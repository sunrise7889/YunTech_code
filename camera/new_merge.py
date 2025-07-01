import cv2
import numpy as np
import pyrealsense2 as rs
import time
import joblib
import pandas as pd
import threading
from collections import deque
from xarm.wrapper import XArmAPI

# === 轉換矩陣模型導入
T = np.load("arm\matrix_5.npy")
# == 用法
#point = np.array(camera+[1]).reshape(4, 1)
#point_arm = T @ point

# === 變數定義 ===
pred = joblib.load("svm_model_no_scaler.pkl")

svm_just_completed = False
is_svm_move = False
selected_roi = None
x0, y0 = 0, 0
drawing = False
template = None
arm_pos = [300, 150]  # 手臂位置模擬
PUCK_RADIUS = 17  # 冰球半徑 (px)
target_width, target_height = 600, 300
px_to_cm = 120 / target_width  # 桌面寬度 120cm 對應 600px
alpha = 0.5
PREV_HANDLE_INDEX = 5  # 取前幾幀握把座標
arm_busy = False            # 控制手臂是否正在移動
last_move_pos = None        # 記錄上一次目標位置
MOVE_THRESHOLD_MM = 5       # 小於此值不移動（避免抖動）
hit_triggered = False  # 防止重複觸發移動
predicted_hit = None
collision_detected = False
prev_ball_speed = 0
COLLISION_DISTANCE = 50  # 握把與球的碰撞距離閾值(px)
SPEED_INCREASE_THRESHOLD = 10  # 球速突然增加的閾值，判斷被撞擊
svm_protection_timer = 0  # 新增：SVM保護計時器
svm_target_reached = False
defense_target = None
SVM_PROTECTION_TIME = 0.01  # SVM完成後保護3秒，避免立即回原點
last_move_reset_timer = 0
POSITION_RESET_TIME = 2.0  # 2秒後重置位置記錄


#Connect ARM and Init
armAPI = XArmAPI('192.168.1.160')
armAPI.clean_error()
armAPI.motion_enable(True)
armAPI.set_mode(0)
armAPI.set_state(0)
armAPI.set_position(x=170, y=0, z=155.5,roll=180, pitch=0, yaw=0,speed=500, acceleration=50000, jerk=100000,wait=False)
print("初始化座標完成")


# 顏色範圍
lower_green = np.array([40, 50, 50])
upper_green = np.array([90, 255, 255])

# 緩衝區
prev_history = deque(maxlen=5)
handle_buffer = []  # 儲存握把座標 buffer
points_2d = []
prev_g_pos = None
prev_time = None
speed_cmps = 0.0
SVMlock =0
# 邊界設定
margin = 5  # px 距離
left_bound = margin
right_bound = target_width - margin
top_bound = margin
bottom_bound = target_height - margin

# 計算有效邊界（考慮球半徑）
effective_left = left_bound + PUCK_RADIUS
effective_right = right_bound - PUCK_RADIUS
effective_top = top_bound + PUCK_RADIUS
effective_bottom = bottom_bound - PUCK_RADIUS

# 線條定義
center_line_x = target_width // 2
defense_line_x = 480  # 防守線靠右側
arm_line = 200  #手臂啟動線

# 速度閾值
SPEED_THRESHOLD = 30  # cm/s


def delayed_unlock():
    global arm_busy
    arm_busy = False

def svm_arm(x, y):
    global arm_busy, last_move_pos, is_svm_move, last_move_reset_timer, svm_target_reached

    # 若手臂仍在移動中，直接略過
    if arm_busy:
        return

    # 相機 → 機械手臂座標轉換
    camera = [x, y, 870]
    point = np.array(camera + [1]).reshape(4, 1)
    arm = T @ point
    arm_xyz = arm[:3].flatten()
    current_time = time.time()
    if last_move_reset_timer > 0 and (current_time - last_move_reset_timer) > POSITION_RESET_TIME:
        last_move_pos = None
        last_move_reset_timer = 0
        
    # 若距離與上一次相近，則不需要移動
    if last_move_pos is not None:
        diff = np.linalg.norm(arm_xyz - last_move_pos)
        if diff < MOVE_THRESHOLD_MM:
            threading.Thread(target=delayed_unlock).start()
            return  # 距離變化太小，不移動
    last_move_pos = arm_xyz  # 更新紀錄
    last_move_reset_timer = current_time
    
    # 啟動新的移動執行緒
    arm_busy = True
    def task():
        global arm_busy, svm_target_reached
        try:
            error_code = armAPI.set_position(*arm_xyz, speed=500, acceleration=50000, jerk=100000, wait=False)
            if error_code != 0:
                print("Error code:", error_code)
                print("State:", armAPI.get_state())
            else:
                svm_target_reached = True
                print("SVM到達目標位置:", arm_xyz)
        finally:
            arm_busy = False
    threading.Thread(target=task).start()
    is_svm_move = True


def get_g_pos(x, y):
    global arm_busy, last_move_pos, last_move_reset_timer

    # 若手臂仍在移動中，直接略過
    if arm_busy:
        return

    # 相機 → 機械手臂座標轉換
    camera = [x * 2, y * 2, 870]
    point = np.array(camera + [1]).reshape(4, 1)
    arm = T @ point
    arm_xyz = arm[:3].flatten()

    # 檢查是否需要重置位置記錄（避免卡住）
    current_time = time.time()
    if last_move_reset_timer > 0 and (current_time - last_move_reset_timer) > POSITION_RESET_TIME:
        last_move_pos = None
        last_move_reset_timer = 0
        print("重置位置記錄")

    # 若距離與上一次相近，則不需要移動
    if last_move_pos is not None:
        diff = np.linalg.norm(arm_xyz - last_move_pos)
        if diff < MOVE_THRESHOLD_MM:
            print(f"距離太小，不移動: {diff:.2f}mm")
            threading.Thread(target=delayed_unlock).start()
            return  # 距離變化太小，不移動
    
    last_move_pos = arm_xyz  # 更新紀錄
    last_move_reset_timer = current_time  # 開始計時

    # 啟動新的移動執行緒
    arm_busy = True
    def task():
        global arm_busy
        try:
            error_code = armAPI.set_position(*arm_xyz, speed=500, acceleration=50000, jerk=100000, wait=False)
            if error_code != 0:
                print("Error code:", error_code)
                print("State:", armAPI.get_state())
            else:
                print("Defense Move complete:", arm_xyz)
        finally:
            arm_busy = False  # 無論成功與否皆解鎖
    threading.Thread(target=task).start()




# def safe_get_depth(depth_frame, x, y):
#     if not depth_frame:
#         return None
#     depth_value = depth_frame.get_distance(x, y)
#     if depth_value == 0 or np.isnan(depth_value):
#         return None
#     return depth_value * 1000  # 轉成 mm


def predict_collision_with_radius(cx, cy, vx, vy):
    """精確的球體邊緣碰撞檢測"""
    t_values = {}
    
    # 計算球邊緣到各邊界的距離
    left_dist = (effective_left - cx) 
    right_dist = (effective_right - cx)
    top_dist = (effective_top - cy)
    bottom_dist = (effective_bottom - cy)
    
    # X軸碰撞檢測 (考慮球半徑)
    if vx > 0:
        t_values['right'] = right_dist / vx if vx != 0 else float('inf')
    elif vx < 0:
        t_values['left'] = left_dist / vx if vx != 0 else float('inf')
    
    # Y軸碰撞檢測 (考慮球半徑)
    if vy > 0:
        t_values['bottom'] = bottom_dist / vy if vy != 0 else float('inf')
    elif vy < 0:
        t_values['top'] = top_dist / vy if vy != 0 else float('inf')
    
    if not t_values:
        return None, None
    
    # 找出最早碰撞的邊界
    collision_boundary = min(t_values, key=t_values.get)
    t_min = t_values[collision_boundary]
    
    # 計算碰撞點（球邊緣接觸邊界時的球心位置）
    collision_x = cx + vx * t_min
    collision_y = cy + vy * t_min
    
    return (collision_x, collision_y), collision_boundary

def calculate_reflection(vx, vy, boundary):
    """反射向量計算"""
    if boundary in ['left', 'right']:
        return -vx, vy  # x方向反向
    elif boundary in ['top', 'bottom']:
        return vx, -vy  # y方向反向
    else:
        return vx, vy

def draw_reflection_path_until_line(start_pos, velocity, target_line_x, bounds, max_bounce=5):
    path = []
    current_pos = np.array(start_pos, dtype=np.float32)
    current_vel = np.array(velocity, dtype=np.float32)
    for _ in range(max_bounce):
        intersection = find_intersection_with_line(current_pos, current_vel, target_line_x)
        if intersection is not None:
            path.append((current_pos.copy(), intersection))
            break
        collision, boundary = predict_collision_with_radius(current_pos[0], current_pos[1], current_vel[0], current_vel[1])
        if collision is None:
            break
        path.append((current_pos.copy(), collision))
        current_pos = np.array(collision)
        current_vel = np.array(calculate_reflection(current_vel[0], current_vel[1], boundary))
    return path

def find_intersection_with_line(start_pos, velocity, line_x):
    """計算軌跡與垂直線的交點"""
    x0, y0 = start_pos
    vx, vy = velocity
    
    if vx == 0:  # 垂直移動，不會與垂直線相交
        return None
    
    t = (line_x - x0) / vx
    if t < 0:  # 只考慮未來的交點
        return None
    
    intersection_y = y0 + vy * t
    
    # 檢查交點是否在有效範圍內
    if effective_top <= intersection_y <= effective_bottom:
        return (line_x, intersection_y)
    return None

# === 滑鼠事件：選角點 ===
def select_corners(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_2d) < 4:
        points_2d.append((x, y))
        print(f"點選角點: ({x}, {y})")

def detect_collision(cx_h, cy_h, cx_g, cy_g, current_speed, prev_speed):
    """檢測握把是否撞擊到球"""
    if cx_h is None or cy_h is None or cx_g is None or cy_g is None:
        return False
    
    # 方法1: 距離檢測
    distance = np.sqrt((cx_h - cx_g)**2 + (cy_h - cy_g)**2)
    distance_collision = distance < COLLISION_DISTANCE
    
    # 方法2: 球速突然增加檢測
    speed_increase = current_speed > prev_speed + SPEED_INCREASE_THRESHOLD
    
    # 方法3: 握把在球的左側且距離很近（符合撞擊方向）
    direction_ok = cx_h < cx_g + 20  # 握把在球左側或稍微右側
    
    collision_result = distance_collision and (speed_increase or direction_ok or current_speed > 5)
    
    if distance < 50:  # 當距離小於50時顯示debug資訊
        print(f"碰撞檢測 - 距離:{distance:.1f}, 速度:{current_speed:.1f}, 增速:{speed_increase}, 方向:{direction_ok}, 結果:{collision_result}")
    return collision_result

# === 滑鼠事件：選握把 ===
def roi_callback(event, x, y, flags, param):
    global x0, y0, drawing, selected_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x0, y0 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = x, y
        selected_roi = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

# === 初始化 RealSense ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
pipeline.start(config)

print("等待 3 秒穩定畫面...")
time.sleep(3)

# === 第一階段: 點選四角 ===
cv2.namedWindow("Select Corners")
cv2.setMouseCallback("Select Corners", select_corners)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    frame = np.asanyarray(color_frame.get_data()).copy()
    show = frame.copy()
    for pt in points_2d:
        cv2.circle(show, pt, 5, (0, 255, 0), -1)
    cv2.imshow("Select Corners", show)
    if len(points_2d) == 4 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("Select Corners")

# === Homography ===
target_pts = np.array([[0, 0], [target_width - 1, 0],
                       [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)
H = cv2.getPerspectiveTransform(np.array(points_2d, dtype=np.float32), target_pts)

# === 第二階段: 選擇握把 ===
cv2.namedWindow("Select Handle")
cv2.setMouseCallback("Select Handle", roi_callback)
print("請在俯視影像上框選握把...")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    frame = np.asanyarray(color_frame.get_data())
    warped = cv2.warpPerspective(frame, H, (target_width, target_height))
    show = warped.copy()

    if selected_roi is not None:
        x, y, w, h = selected_roi
        cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 255), 2)
        template = warped[y:y+h, x:x+w].copy()
        print("握把選取完成，開始追蹤...")
        break

    cv2.imshow("Select Handle", show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

cv2.destroyWindow("Select Handle")

# === 第三階段: 開始追蹤 ===
cv2.namedWindow("Tracking")

try:
    while True:
        frames = pipeline.wait_for_frames()
        frame = np.asanyarray(frames.get_color_frame().get_data())
        warped = cv2.warpPerspective(frame, H, (target_width, target_height))
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

        # === 顯示邊界和線條 ===
        cv2.rectangle(warped, (left_bound, top_bound), (right_bound, bottom_bound), (0, 255, 255), 2)
        cv2.line(warped, (center_line_x, 0), (center_line_x, target_height), (0, 0, 255), 2)  # 中線
        cv2.line(warped, (defense_line_x, 0), (defense_line_x, target_height), (0, 0, 255), 2)  # 防守線 (紅色)
        # 畫出右邊界上半段與下半段區域（SVM 預測用）
        cv2.line(warped, (right_bound, top_bound), (right_bound, target_height // 2), (255, 0, 255), 4)  # 上半段：紫色 (0)
        cv2.line(warped, (right_bound, target_height // 2), (right_bound, bottom_bound), (255, 255, 0), 4)  # 下半段：青色 (1)

        # === 冰球追蹤 ===
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cx_g, cy_g = None, None
        vx, vy = 0, 0
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx_g = M["m10"] / M["m00"]
                cy_g = M["m01"] / M["m00"]
                
                # 繪製冰球（球心+半徑範圍）
                cv2.circle(warped, (int(cx_g), int(cy_g)), int(PUCK_RADIUS), (255, 0, 255), 1)
                cv2.circle(warped, (int(cx_g), int(cy_g)), 5, (255, 0, 0), -1)
                
                # 更新歷史座標
                prev_history.append((cx_g, cy_g))
                
                # 計算速度
                now = time.time()
                if prev_g_pos is not None and prev_time is not None:
                    dx = cx_g - prev_g_pos[0]
                    dy = cy_g - prev_g_pos[1]
                    dt = now - prev_time
                    
                    if dt > 0:
                        dist_px = np.sqrt(dx**2 + dy**2)
                        if dist_px < 0.5:
                            new_speed = 0
                        else:
                            dist_cm = dist_px * px_to_cm
                            new_speed = dist_cm / dt
                        
                        speed_cmps = alpha * speed_cmps + (1 - alpha) * new_speed
                        speed_cmps = 0 if (np.isnan(speed_cmps) or speed_cmps > 1000 or speed_cmps < 0.3) else speed_cmps
                
                # 計算平滑速度向量
                if len(prev_history) >= 5:
                    sum_vx, sum_vy = 0, 0
                    for i in range(1, len(prev_history)):
                        dx = prev_history[i][0] - prev_history[i - 1][0]
                        dy = prev_history[i][1] - prev_history[i - 1][1]
                        sum_vx += dx
                        sum_vy += dy
                    vx = sum_vx / (len(prev_history) - 1)
                    vy = sum_vy / (len(prev_history) - 1)

                
                prev_g_pos = (cx_g, cy_g)
                prev_time = now
        
        # === 握把追蹤 ===
        cx_h, cy_h = None, None
        prev_handle = None
        
        if selected_roi is not None and template is not None:
            res = cv2.matchTemplate(warped, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.5:
                top_left = max_loc
                bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
                cv2.rectangle(warped, top_left, bottom_right, (0, 0, 255), 2)
                cx_h = top_left[0] + template.shape[1] // 2
                cy_h = top_left[1] + template.shape[0] // 2
                cv2.circle(warped, (cx_h, cy_h), 5, (255, 0, 0), -1)

                handle_buffer.append((cx_h, cy_h))
                if len(handle_buffer) > PREV_HANDLE_INDEX:
                    prev_handle = handle_buffer[-PREV_HANDLE_INDEX - 1]
        
        # === SVM預測 ===
        if (cx_h is not None and cy_h is not None and prev_handle is not None and SVMlock == 0):
            dx = cx_h - prev_handle[0]
            dy = cy_h - prev_handle[1]
            dist = np.hypot(dx, dy)

            if dx > 10 and dist > 15:  # 握把往右揮動且有夠大幅度
                SVMlock = 1  # 先鎖住，防止重複觸發
                svm_just_completed = True
                svm_protection_timer = time.time()
                
                input_data = pd.DataFrame([[cx_h, cy_h, prev_handle[0], prev_handle[1], cx_g or 0, cy_g or 0]],
                    columns=["hand_x", "hand_y", "prev_hand_x", "prev_hand_y", "ball_x", "ball_y"])
                prediction = pred.predict(input_data)

                if prediction == 0:
                    svm_arm(1000,122)                   
                    print("SVM 預測: 上半")
                elif prediction == 1:
                    svm_arm(1000,480)
                    print("SVM 預測: 下半")
                
                
        # 若手把回到左側，解除鎖定
        if cx_h is not None and cx_h < 150 and not is_svm_move and svm_target_reached:
            SVMlock = 0
            svm_just_completed = False
            svm_target_reached = False
            last_move_pos = None
            print("SVM完成並解鎖")

        # === 擊球後預測反射路徑並畫線 ===
        if vx > 1:
            predicted_path = draw_reflection_path_until_line(
                start_pos=(cx_g, cy_g),
                velocity=(vx, vy),
                target_line_x = defense_line_x,
                bounds=(left_bound, right_bound, top_bound, bottom_bound)
            )
            for seg_start, seg_end in predicted_path:
                cv2.line(warped, tuple(map(int, seg_start)), tuple(map(int, seg_end)), (0, 255, 255), 2)
                cv2.circle(warped, tuple(map(int, seg_end)), 4, (0, 255, 255), -1)
            if predicted_path:
                predicted_hit = predicted_path[-1][1]
             
        current_collision = detect_collision(cx_h, cy_h, cx_g, cy_g, speed_cmps, prev_ball_speed)   
        
        # if current_collision and not collision_detected:
        #     collision_detected = True
        #     print("🔥 撞擊檢測觸發！")
            
        #     # 立即計算防守線交點並記錄
        #     if predicted_hit is not None and abs(predicted_hit[0] - defense_line_x) < 5:
        #         defense_target = predicted_hit
        #         print(f"記錄防守目標點: ({defense_target[0]:.1f}, {defense_target[1]:.1f})")
                
        #         # 如果SVM已經在移動，強制提前準備防守
        #         if is_svm_move and not hit_triggered:
        #             svm_target_reached = True  # 強制標記可以防守
        #             print("SVM移動中，提前準備防守")
        # elif not current_collision:
        #     # 當沒有撞擊時，可以重置撞擊狀態（但不立即，避免抖動）
        #     pass
        if current_collision and not collision_detected:
            collision_detected = True
            print("🔥 撞擊檢測觸發！")
            
            # 立即計算防守線交點並記錄
            if predicted_hit is not None and abs(predicted_hit[0] - defense_line_x) < 5:
                defense_target = predicted_hit
                print(f"記錄防守目標點: ({defense_target[0]:.1f}, {defense_target[1]:.1f})")
                
                # 立即開始防守，不管SVM狀態
                if not hit_triggered:
                    x_cam, y_cam = int(defense_target[0]), int(defense_target[1])
                    get_g_pos(x_cam, y_cam)
                    hit_triggered = True
                    is_svm_move = False  # 強制結束SVM
                    print(f"立即防守移動: ({x_cam}, {y_cam})")
                    cv2.circle(warped, (x_cam, y_cam), 8, (0, 255, 0), -1)
        elif not current_collision:
            # 當沒有撞擊時，可以重置撞擊狀態（但不立即，避免抖動）
            pass
        # 更新前一幀球速
        prev_ball_speed = speed_cmps
        
        #透過速度判別看是否要防守模式(堅守洞口)
        if speed_cmps <= 200:
            if is_svm_move:
                # 檢查是否可以提前開始防守
                if defense_target is not None and svm_target_reached and not hit_triggered:
                    # 不等SVM完全結束，立即開始防守移動
                    x_cam, y_cam = int(defense_target[0]), int(defense_target[1])
                    get_g_pos(x_cam, y_cam)
                    hit_triggered = True
                    is_svm_move = False  # 結束SVM模式，進入防守模式
                    print(f"提前防守移動到目標點: ({x_cam}, {y_cam})")
                    cv2.circle(warped, (x_cam, y_cam), 8, (0, 255, 0), -1)
                
                # SVM移動中，等待完成
                elif not arm_busy:
                    is_svm_move = False
                    print("SVM移動完成")
                    
                    # 如果有記錄的防守目標點，立即移動實作
                    if defense_target is not None and not hit_triggered:
                        x_cam, y_cam = int(defense_target[0]), int(defense_target[1])
                        get_g_pos(x_cam, y_cam)
                        hit_triggered = True
                        print(f"移動到防守目標點: ({x_cam}, {y_cam})")
                        cv2.circle(warped, (x_cam, y_cam), 8, (0, 255, 0), -1)
            else:
                # 撞擊檢測到後，移動到預測落點（只觸發一次）
                # 但不要在SVM保護期內觸發
                current_time = time.time()                
                ball_moving_right = vx > 0.5  # 球向右移動
                ball_in_right_area = cx_g is not None and cx_g > center_line_x  # 球在右半邊

                svm_protection_active = (SVMlock == 1 and not svm_target_reached)

                
                # 當球回到左邊，解除觸發鎖
                if cx_g is not None and cx_g < arm_line:
                    hit_triggered = False
                    collision_detected = False
                    predicted_hit = None
                    defense_target = None
                
                # 修改回原點條件：更嚴格的檢查
                svm_protection_expired = (current_time - svm_protection_timer) > SVM_PROTECTION_TIME
                
                if (cx_g is not None and cx_g < arm_line and not arm_busy 
                    and not hit_triggered and SVMlock == 0 
                    and not svm_just_completed and svm_protection_expired
                    and not is_svm_move):
                    # 回歸初始點前重置位置記錄
                    last_move_pos = None
                    armAPI.set_position(x=170, y=0, z=155.5, speed=500, 
                                      acceleration=50000, jerk=100000, wait=False)
                
        elif speed_cmps > 200:
            if not arm_busy and not is_svm_move:
                arm_busy = True
                def task():
                    global arm_busy, is_svm_move, collision_detected, svm_just_completed, SVMlock, last_move_pos
                    try:
                        armAPI.set_position(x=170, y=0, z=155.5, speed=500, 
                                          acceleration=50000, jerk=100000, wait=False)
                        print("球速過快，退回中心防守")
                        is_svm_move = False
                        collision_detected = False
                        svm_just_completed = False
                        SVMlock = 0
                        last_move_pos = None  # 重置位置記錄
                    finally:
                        arm_busy = False
                threading.Thread(target=task).start()

        # === 顯示速度 ===
        cv2.putText(warped, f"Speed: {speed_cmps:.1f} cm/s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Tracking", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()