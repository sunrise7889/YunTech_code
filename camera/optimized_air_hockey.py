import csv
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import joblib
import pandas as pd
import threading
from collections import deque
from xarm.wrapper import XArmAPI

class PerformanceTracker:
    def __init__(self, csv_filename="performance.csv"):
        self.csv_filename = csv_filename
        self.run_number = 0
        self.current_run = {}
        self.run_active = False
        self.setup_csv()
        
    def setup_csv(self):
        """初始化CSV檔案，如果不存在則建立"""
        file_exists = os.path.isfile(self.csv_filename)
        
        if not file_exists:
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                headers = ['run_id', 'svm_prediction', 'max_speed', 'hit_result']
                writer.writerow(headers)
                print(f"已建立性能記錄檔案: {self.csv_filename}")
        else:
            # 讀取現有檔案獲取最後的流水號
            try:
                with open(self.csv_filename, 'r', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)  # 跳過標題
                    rows = list(reader)
                    if rows:
                        self.run_number = int(rows[-1][0])
            except:
                self.run_number = 0
    
    def start_run(self, svm_prediction):
        """開始新的run
        Args:
            svm_prediction: 0=上半, 1=下半, -1=無SVM預測
        """
        if self.run_active:
            # 如果上一個run還沒結束，先記錄為失敗
            self.end_run(hit_result=0)
        
        self.run_number += 1
        self.run_active = True
        self.current_run = {
            'run_id': self.run_number,
            'svm_prediction': svm_prediction,
            'max_speed': 0,
            'hit_result': 0
        }
        print(f"開始 Run {self.run_number}, SVM預測: {svm_prediction}")
    
    def update_speed(self, speed):
        """更新當前run的最高球速"""
        if self.run_active and speed > self.current_run['max_speed']:
            self.current_run['max_speed'] = round(speed, 1)
    
    def end_run(self, hit_result):
        """結束當前run
        Args:
            hit_result: 1=打到, 0=沒打到
        """
        if not self.run_active:
            return
        
        self.current_run['hit_result'] = hit_result
        
        # 寫入CSV
        with open(self.csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            row = [
                self.current_run['run_id'],
                self.current_run['svm_prediction'],
                self.current_run['max_speed'],
                self.current_run['hit_result']
            ]
            writer.writerow(row)
        
        print(f"Run {self.current_run['run_id']} 完成 - "
              f"SVM: {self.current_run['svm_prediction']}, "
              f"速度: {self.current_run['max_speed']}, "
              f"結果: {'打到' if hit_result else '沒打到'}")
        
        # 清空並結束run
        self.run_active = False
        self.current_run = {}


class AirHockeyBot:
    def __init__(self):
        # === 核心參數設定 ===
        self.T = np.load("arm/matrix_5.npy") #座標轉換矩陣
        self.pred = joblib.load("svm_model_no_scaler.pkl") #SVM預測模型
        
        # 追蹤參數
        self.PUCK_RADIUS = 17
        self.target_width, self.target_height = 600, 300
        self.px_to_cm = 120 / self.target_width
        self.alpha = 0.5
        self.PREV_HANDLE_INDEX = 5
        self.MOVE_THRESHOLD_MM = 5
        self.COLLISION_DISTANCE = 50
        self.SPEED_INCREASE_THRESHOLD = 10
        self.SVM_PROTECTION_TIME = 0.01
        self.POSITION_RESET_TIME = 2.0
        self.SPEED_THRESHOLD = 30
        
        # 邊界設定
        margin = 5
        self.left_bound = margin
        self.right_bound = self.target_width - margin
        self.top_bound = margin
        self.bottom_bound = self.target_height - margin
        
        # 有效邊界（考慮球半徑）
        self.effective_left = self.left_bound + self.PUCK_RADIUS
        self.effective_right = self.right_bound - self.PUCK_RADIUS
        self.effective_top = self.top_bound + self.PUCK_RADIUS
        self.effective_bottom = self.bottom_bound - self.PUCK_RADIUS
        
        # 線條定義
        self.center_line_x = self.target_width // 2
        self.defense_line_x = 480
        self.arm_line = 200
        
        # 顏色範圍
        self.lower_green = np.array([40, 50, 50])
        self.upper_green = np.array([90, 255, 255])
        
        # 狀態變數
        self.arm_busy = False
        self.is_svm_move = False
        self.svm_just_completed = False
        self.svm_target_reached = False
        self.hit_triggered = False
        self.collision_detected = False
        self.SVMlock = 0
        self.last_move_pos = None
        self.last_move_reset_timer = 0
        self.svm_protection_timer = 0
        self.defense_target = None
        self.predicted_hit = None
        
        # 緩衝區和歷史記錄
        self.prev_history = deque(maxlen=5)
        self.handle_buffer = []
        self.prev_g_pos = None
        self.prev_time = None
        self.speed_cmps = 0.0
        self.prev_ball_speed = 0
        
        # UI相關
        self.points_2d = []
        self.selected_roi = None
        self.x0, self.y0 = 0, 0
        self.drawing = False
        self.template = None
        
        # 初始化手臂
        self.init_arm()
        
        # 初始化攝影機
        self.init_camera()
        self.performance_tracker = PerformanceTracker()

    def init_arm(self):
        """初始化機械手臂"""
        self.armAPI = XArmAPI('192.168.1.160')
        self.armAPI.clean_error()
        self.armAPI.motion_enable(True)
        self.armAPI.set_mode(0)
        self.armAPI.set_state(0)
        self.armAPI.set_position(x=170, y=0, z=155.5, roll=180, pitch=0, yaw=0,
                               speed=500, acceleration=50000, jerk=100000, wait=False)
        print("機械手臂初始化完成")
    
    def init_camera(self):
        """初始化RealSense攝影機"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
        self.pipeline.start(config)
        print("攝影機初始化完成，等待3秒穩定...")
        time.sleep(3)
    
    def select_corners_callback(self, event, x, y, flags, param):
        """角點選擇回調函數"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points_2d) < 4:
            self.points_2d.append((x, y))
            print(f"點選角點: ({x}, {y})")
    
    def roi_callback(self, event, x, y, flags, param):
        """ROI選擇回調函數"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x0, self.y0 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = x, y
            self.selected_roi = (min(self.x0, x1), min(self.y0, y1), 
                               abs(x1 - self.x0), abs(y1 - self.y0))
    
    def setup_tracking(self):
        """設定追蹤參數（選擇角點和握把）"""
        # 第一階段：選擇角點
        cv2.namedWindow("Select Corners")
        cv2.setMouseCallback("Select Corners", self.select_corners_callback)
        
        print("請點選四個角點...")
        while len(self.points_2d) < 4:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            frame = np.asanyarray(color_frame.get_data())
            show = frame.copy()
            
            for pt in self.points_2d:
                cv2.circle(show, pt, 5, (0, 255, 0), -1)
            
            cv2.imshow("Select Corners", show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow("Select Corners")
        
        # 計算單應性矩陣
        target_pts = np.array([[0, 0], [self.target_width - 1, 0],
                             [self.target_width - 1, self.target_height - 1], 
                             [0, self.target_height - 1]], dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(np.array(self.points_2d, dtype=np.float32), target_pts)
        
        # 第二階段：選擇握把
        cv2.namedWindow("Select Handle")
        cv2.setMouseCallback("Select Handle", self.roi_callback)
        print("請框選握把...")
        
        while self.selected_roi is None:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())
            warped = cv2.warpPerspective(frame, self.H, (self.target_width, self.target_height))
            
            cv2.imshow("Select Handle", warped)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        
        # 儲存握把模板
        x, y, w, h = self.selected_roi
        self.template = warped[y:y+h, x:x+w].copy()
        cv2.destroyWindow("Select Handle")
        print("設定完成，開始追蹤...")
        return True
    
    def delayed_unlock(self):
        """延遲解鎖手臂"""
        self.arm_busy = False
    
    def move_arm_async(self, x, y, z=155.5, is_svm=False):
        """非同步移動手臂"""
        if self.arm_busy:
            return False
        
        # 座標轉換
        if is_svm:
            camera = [x, y, 870]
        else:
            camera = [x * 2, y * 2, 870]
        
        point = np.array(camera + [1]).reshape(4, 1)
        arm = self.T @ point
        arm_xyz = arm[:3].flatten()
        
        # 檢查移動距離
        current_time = time.time()
        if self.last_move_reset_timer > 0 and (current_time - self.last_move_reset_timer) > self.POSITION_RESET_TIME:
            self.last_move_pos = None
            self.last_move_reset_timer = 0
        
        if self.last_move_pos is not None:
            diff = np.linalg.norm(arm_xyz - self.last_move_pos)
            if diff < self.MOVE_THRESHOLD_MM:
                threading.Thread(target=self.delayed_unlock).start()
                return False
        
        self.last_move_pos = arm_xyz
        self.last_move_reset_timer = current_time
        self.arm_busy = True
        
        def move_task():
            try:
                error_code = self.armAPI.set_position(*arm_xyz, speed=500, 
                                                    acceleration=50000, jerk=100000, wait=False)
                if error_code != 0:
                    print("Error code:", error_code)
                else:
                    if is_svm:
                        self.svm_target_reached = True
                        print(f"SVM移動完成: {arm_xyz}")
                    else:
                        print(f"防守移動完成: {arm_xyz}")
            finally:
                self.arm_busy = False
        
        threading.Thread(target=move_task).start()
        return True
    
    def predict_collision_with_radius(self, cx, cy, vx, vy):
        """精確的球體邊緣碰撞檢測"""
        t_values = {}
        
        # 計算到各邊界的時間
        if vx > 0:
            t_values['right'] = (self.effective_right - cx) / vx
        elif vx < 0:
            t_values['left'] = (self.effective_left - cx) / vx
        
        if vy > 0:
            t_values['bottom'] = (self.effective_bottom - cy) / vy
        elif vy < 0:
            t_values['top'] = (self.effective_top - cy) / vy
        
        if not t_values:
            return None, None
        
        # 找出最早碰撞
        collision_boundary = min(t_values, key=t_values.get)
        t_min = t_values[collision_boundary]
        
        collision_x = cx + vx * t_min
        collision_y = cy + vy * t_min
        
        return (collision_x, collision_y), collision_boundary
    
    def calculate_reflection(self, vx, vy, boundary):
        """計算反射向量"""
        if boundary in ['left', 'right']:
            return -vx, vy
        elif boundary in ['top', 'bottom']:
            return vx, -vy
        return vx, vy
    
    def find_intersection_with_line(self, start_pos, velocity, line_x):
        """計算軌跡與垂直線的交點"""
        x0, y0 = start_pos
        vx, vy = velocity
        
        if vx == 0:
            return None
        
        t = (line_x - x0) / vx
        if t < 0:
            return None
        
        intersection_y = y0 + vy * t
        
        if self.effective_top <= intersection_y <= self.effective_bottom:
            return (line_x, intersection_y)
        return None
    
    def draw_reflection_path(self, start_pos, velocity, max_bounce=5):
        """繪製反射路徑"""
        path = []
        current_pos = np.array(start_pos, dtype=np.float32)
        current_vel = np.array(velocity, dtype=np.float32)
        
        for _ in range(max_bounce):
            # 檢查是否與防守線相交
            intersection = self.find_intersection_with_line(current_pos, current_vel, self.defense_line_x)
            if intersection is not None:
                path.append((current_pos.copy(), intersection))
                break
            
            # 計算碰撞點
            collision, boundary = self.predict_collision_with_radius(
                current_pos[0], current_pos[1], current_vel[0], current_vel[1])
            if collision is None:
                break
            
            path.append((current_pos.copy(), collision))
            current_pos = np.array(collision)
            current_vel = np.array(self.calculate_reflection(current_vel[0], current_vel[1], boundary))
        
        return path
    
    def detect_collision(self, cx_h, cy_h, cx_g, cy_g, current_speed, prev_speed):
        """檢測握把與球的碰撞"""
        if None in [cx_h, cy_h, cx_g, cy_g]:
            return False
        
        distance = np.sqrt((cx_h - cx_g)**2 + (cy_h - cy_g)**2)
        distance_collision = distance < self.COLLISION_DISTANCE
        speed_increase = current_speed > prev_speed + self.SPEED_INCREASE_THRESHOLD
        direction_ok = cx_h < cx_g + 20
        
        return distance_collision and (speed_increase or direction_ok or current_speed > 5)
    
    def process_frame(self, warped):
        """處理單一幀的主要邏輯"""
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        
        # 繪製邊界和線條
        self.draw_boundaries(warped)
        
        # 追蹤冰球
        cx_g, cy_g, vx, vy = self.track_puck(warped, hsv)
        
        # 追蹤握把
        cx_h, cy_h, prev_handle = self.track_handle(warped)
        
        # SVM預測
        self.process_svm_prediction(cx_h, cy_h, prev_handle, cx_g, cy_g)
        
        # 碰撞檢測和防守邏輯
        predicted_path = self.process_collision_and_defense(warped, cx_g, cy_g, vx, vy, cx_h, cy_h)
        
        # 繪製預測路徑
        if predicted_path:
            self.draw_prediction_path(warped, predicted_path)
        
        # 更新狀態
        self.update_game_state(cx_g, cy_g, vx, vy)
        
        # 顯示資訊
        cv2.putText(warped, f"Speed: {self.speed_cmps:.1f} cm/s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return warped
    
    def draw_boundaries(self, warped):
        """繪製邊界和線條"""
        # 邊界框
        cv2.rectangle(warped, (self.left_bound, self.top_bound), 
                     (self.right_bound, self.bottom_bound), (0, 255, 255), 2)
        
        # 中線
        cv2.line(warped, (self.center_line_x, 0), (self.center_line_x, self.target_height), (0, 0, 255), 2)
        
        # 防守線
        cv2.line(warped, (self.defense_line_x, 0), (self.defense_line_x, self.target_height), (0, 0, 255), 2)
        
        # SVM區域標示
        cv2.line(warped, (self.right_bound, self.top_bound), 
                (self.right_bound, self.target_height // 2), (255, 0, 255), 4)
        cv2.line(warped, (self.right_bound, self.target_height // 2), 
                (self.right_bound, self.bottom_bound), (255, 255, 0), 4)
    
    def track_puck(self, warped, hsv):
        """追蹤冰球"""
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cx_g, cy_g, vx, vy = None, None, 0, 0
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx_g = M["m10"] / M["m00"]
                cy_g = M["m01"] / M["m00"]
                
                # 繪製冰球
                cv2.circle(warped, (int(cx_g), int(cy_g)), self.PUCK_RADIUS, (255, 0, 255), 1)
                cv2.circle(warped, (int(cx_g), int(cy_g)), 5, (255, 0, 0), -1)
                
                # 更新速度計算
                self.update_puck_velocity(cx_g, cy_g)
                
                # 計算速度向量
                if len(self.prev_history) >= 5:
                    vx, vy = self.calculate_velocity_vector()
        
        return cx_g, cy_g, vx, vy
    
    def update_puck_velocity(self, cx_g, cy_g):
        """更新冰球速度"""
        self.prev_history.append((cx_g, cy_g))
        
        now = time.time()
        if self.prev_g_pos is not None and self.prev_time is not None:
            dx = cx_g - self.prev_g_pos[0]
            dy = cy_g - self.prev_g_pos[1]
            dt = now - self.prev_time
            
            if dt > 0:
                dist_px = np.sqrt(dx**2 + dy**2)
                if dist_px < 0.5:
                    new_speed = 0
                else:
                    dist_cm = dist_px * self.px_to_cm
                    new_speed = dist_cm / dt
                
                self.speed_cmps = self.alpha * self.speed_cmps + (1 - self.alpha) * new_speed
                if np.isnan(self.speed_cmps) or self.speed_cmps > 1000 or self.speed_cmps < 0.3:
                    self.speed_cmps = 0
        
        self.prev_g_pos = (cx_g, cy_g)
        self.prev_time = now
        self.performance_tracker.update_speed(self.speed_cmps)

    def calculate_velocity_vector(self):
        """計算速度向量"""
        sum_vx, sum_vy = 0, 0
        for i in range(1, len(self.prev_history)):
            dx = self.prev_history[i][0] - self.prev_history[i - 1][0]
            dy = self.prev_history[i][1] - self.prev_history[i - 1][1]
            sum_vx += dx
            sum_vy += dy
        return sum_vx / (len(self.prev_history) - 1), sum_vy / (len(self.prev_history) - 1)
    
    def track_handle(self, warped):
        """追蹤握把"""
        cx_h, cy_h, prev_handle = None, None, None
        
        if self.selected_roi is not None and self.template is not None:
            res = cv2.matchTemplate(warped, self.template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > 0.5:
                top_left = max_loc
                bottom_right = (top_left[0] + self.template.shape[1], 
                              top_left[1] + self.template.shape[0])
                cv2.rectangle(warped, top_left, bottom_right, (0, 0, 255), 2)
                
                cx_h = top_left[0] + self.template.shape[1] // 2
                cy_h = top_left[1] + self.template.shape[0] // 2
                cv2.circle(warped, (cx_h, cy_h), 5, (255, 0, 0), -1)
                
                self.handle_buffer.append((cx_h, cy_h))
                if len(self.handle_buffer) > self.PREV_HANDLE_INDEX:
                    prev_handle = self.handle_buffer[-self.PREV_HANDLE_INDEX - 1]
        
        return cx_h, cy_h, prev_handle
    
    def process_svm_prediction(self, cx_h, cy_h, prev_handle, cx_g, cy_g):
        """處理SVM預測邏輯"""
        if (cx_h is not None and cy_h is not None and prev_handle is not None and self.SVMlock == 0):
            dx = cx_h - prev_handle[0]
            dy = cy_h - prev_handle[1]
            dist = np.hypot(dx, dy)
            
            if dx > 10 and dist > 15:  # 握把往右揮動
                self.SVMlock = 1
                self.svm_just_completed = True
                self.svm_protection_timer = time.time()
                
                input_data = pd.DataFrame([[cx_h, cy_h, prev_handle[0], prev_handle[1], cx_g or 0, cy_g or 0]],
                    columns=["hand_x", "hand_y", "prev_hand_x", "prev_hand_y", "ball_x", "ball_y"])
                prediction = self.pred.predict(input_data)
                self.performance_tracker.start_run(svm_prediction=int(prediction[0]))

                if prediction == 0:
                    self.move_arm_async(1000, 122, is_svm=True)
                    print("SVM 預測: 上半")
                elif prediction == 1:
                    self.move_arm_async(1000, 480, is_svm=True)
                    print("SVM 預測: 下半")
                
                self.is_svm_move = True
        
        # 解除SVM鎖定
        if (cx_h is not None and cx_h < 150 and not self.is_svm_move and self.svm_target_reached):
            self.SVMlock = 0
            self.svm_just_completed = False
            self.svm_target_reached = False
            self.last_move_pos = None
            print("SVM完成並解鎖")
    
    def process_collision_and_defense(self, warped, cx_g, cy_g, vx, vy, cx_h, cy_h):
        """處理碰撞檢測和防守邏輯"""
        predicted_path = None
        
        # 計算預測路徑
        if vx > 1:
            predicted_path = self.draw_reflection_path((cx_g, cy_g), (vx, vy))
            if predicted_path:
                self.predicted_hit = predicted_path[-1][1]
        
        # 碰撞檢測
        current_collision = self.detect_collision(cx_h, cy_h, cx_g, cy_g, self.speed_cmps, self.prev_ball_speed)
        
        if current_collision and not self.collision_detected:
            self.collision_detected = True
            print("🔥 撞擊檢測觸發！")
            
            if self.predicted_hit is not None and abs(self.predicted_hit[0] - self.defense_line_x) < 30:
                self.defense_target = self.predicted_hit
                if not self.hit_triggered:
                    self.move_arm_async(int(self.defense_target[0]), int(self.defense_target[1]))
                    self.hit_triggered = True
                    self.is_svm_move = False
                    cv2.circle(warped, (int(self.defense_target[0]), int(self.defense_target[1])), 8, (0, 255, 0), -1)
        
        # 預防性防守
        if (cx_g is not None and not self.hit_triggered and not self.is_svm_move and 
            cx_g > (self.arm_line + self.center_line_x) / 2 and vx > 0.5 and 
            self.speed_cmps > 5 and not self.collision_detected):
            
            intersection = self.find_intersection_with_line((cx_g, cy_g), (vx, vy), self.defense_line_x)
            if intersection is not None:
                self.defense_target = intersection
                self.move_arm_async(int(self.defense_target[0]), int(self.defense_target[1]))
                self.hit_triggered = True
                cv2.circle(warped, (int(self.defense_target[0]), int(self.defense_target[1])), 8, (0, 255, 0), -1)
        
        self.prev_ball_speed = self.speed_cmps
        return predicted_path
    
    def draw_prediction_path(self, warped, predicted_path):
        """繪製預測路徑"""
        for seg_start, seg_end in predicted_path:
            cv2.line(warped, tuple(map(int, seg_start)), tuple(map(int, seg_end)), (0, 255, 255), 2)
            cv2.circle(warped, tuple(map(int, seg_end)), 4, (0, 255, 255), -1)
    
    def update_game_state(self, cx_g, cy_g, vx, vy):
        """更新遊戲狀態"""
        current_time = time.time()
        
        # 球回到左邊時重置狀態
        if cx_g is not None and cx_g < self.arm_line:
            self.hit_triggered = False
            self.collision_detected = False
            self.predicted_hit = None
            self.defense_target = None
        
        # 根據球速決定防守策略
        if self.speed_cmps <= 200:
            self._handle_normal_speed_defense(current_time)
        else:
            if self.performance_tracker.run_active:
                self.performance_tracker.end_run(hit_result=0)
            self._handle_high_speed_defense()
        
        # 回歸原點條件
        svm_protection_expired = (current_time - self.svm_protection_timer) > self.SVM_PROTECTION_TIME
        if (cx_g is not None and cx_g < self.arm_line and not self.arm_busy 
            and not self.hit_triggered and self.SVMlock == 0 
            and not self.svm_just_completed and svm_protection_expired
            and not self.is_svm_move):
            self.last_move_pos = None
            self.armAPI.set_position(x=170, y=0, z=155.5, speed=500, 
                                   acceleration=50000, jerk=100000, wait=False)
    
    def _handle_normal_speed_defense(self, current_time):
        """處理正常速度下的防守"""
        if self.is_svm_move:
            if self.defense_target is not None and self.svm_target_reached and not self.hit_triggered:
                self.move_arm_async(int(self.defense_target[0]), int(self.defense_target[1]))
                self.hit_triggered = True
                self.is_svm_move = False
            elif not self.arm_busy:
                self.is_svm_move = False
                if self.defense_target is not None and not self.hit_triggered:
                    self.move_arm_async(int(self.defense_target[0]), int(self.defense_target[1]))
                    self.hit_triggered = True
    
    def _handle_high_speed_defense(self):
        """處理高速球的防守"""
        if not self.arm_busy and not self.is_svm_move:
            self.arm_busy = True
            def reset_task():
                try:
                    self.armAPI.set_position(x=170, y=0, z=155.5, speed=500, 
                                           acceleration=50000, jerk=100000, wait=False)
                    print("球速過快，退回中心防守")
                    self.is_svm_move = False
                    self.collision_detected = False
                    self.svm_just_completed = False
                    self.SVMlock = 0
                    self.last_move_pos = None
                finally:
                    self.arm_busy = False
            threading.Thread(target=reset_task).start()
    
    def run(self):
        """主執行迴圈"""
        if not self.setup_tracking():
            return
        
        cv2.namedWindow("Tracking")
        
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                frame = np.asanyarray(frames.get_color_frame().get_data())
                warped = cv2.warpPerspective(frame, self.H, (self.target_width, self.target_height))
                
                # 處理當前幀
                processed_frame = self.process_frame(warped)
                
                cv2.imshow("Tracking", processed_frame)
                
                # 檢查退出條件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理資源"""
        try:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("程式正常結束")
        except Exception as e:
            print(f"清理時發生錯誤: {e}")

# === 主程式入口 ===
if __name__ == "__main__":
    try:
        bot = AirHockeyBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n程式被使用者中斷")
    except Exception as e:
        print(f"程式執行錯誤: {e}")
        import traceback
        traceback.print_exc()