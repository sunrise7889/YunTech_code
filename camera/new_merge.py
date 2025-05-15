import cv2
import numpy as np
import pyrealsense2 as rs
import time
import joblib
import pandas as pd
from collections import deque

# === 變數定義 ===
pred = joblib.load("svm_model_no_scaler.pkl")

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
attack_line_x  = 380  # 攻擊線稍微偏中右

# 速度閾值
SPEED_THRESHOLD = 30  # cm/s

# === 函數定義 ===
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
        cv2.line(warped, (attack_line_x, 0), (attack_line_x, target_height), (255, 0, 0), 2)  # 攻擊線 (藍色)
        cv2.line(warped, (defense_line_x, 0), (defense_line_x, target_height), (0, 0, 255), 2)  # 防守線 (紅色)
        
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
                
                # 計算速度向量
                if len(prev_history) >= 5:
                    prev_cx_g, prev_cy_g = prev_history[-5]
                    vx = cx_g - prev_cx_g
                    vy = cy_g - prev_cy_g
                
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
        if (cx_h is not None and cy_h is not None and prev_handle is not None and 
            cx_g is not None and cy_g is not None and vx > 0):
            prev_x, prev_y = prev_handle
            input_data = pd.DataFrame([[cx_h, cy_h, prev_x, prev_y, cx_g, cy_g]],
                columns=["hand_x", "hand_y", "prev_hand_x", "prev_hand_y", "ball_x", "ball_y"])
            prediction = pred.predict(input_data)
            
            # 根據預測結果決定擊球線
            target_line_x = attack_line_x if prediction == 0 else defense_line_x
            
            # 計算與目標線的交點
            hit_point = find_intersection_with_line((cx_g, cy_g), (vx, vy), target_line_x)
            
            if hit_point:
                # 根據速度決定模式
                if speed_cmps > SPEED_THRESHOLD:
                    # 防守模式：使用防守線
                    hit_point = find_intersection_with_line((cx_g, cy_g), (vx, vy), defense_line_x)
                    cv2.circle(warped, (int(hit_point[0]), int(hit_point[1])), 8, (0, 0, 255), -1)
                    cv2.putText(warped, "Defense Mode", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # 攻擊模式：使用攻擊線
                    hit_point = find_intersection_with_line((cx_g, cy_g), (vx, vy), attack_line_x)
                    cv2.circle(warped, (int(hit_point[0]), int(hit_point[1])), 8, (255, 0, 0), -1)
                    cv2.putText(warped, "Attack Mode", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # 移動手臂到擊球點
                arm_pos = [int(hit_point[0]), int(hit_point[1])]
                
                # 繪製預測軌跡
                cv2.line(warped, (int(cx_g), int(cy_g)), 
                        (int(hit_point[0]), int(hit_point[1])), (0, 255, 0), 2)
        
        # === 顯示手臂位置 ===
        cv2.circle(warped, tuple(arm_pos), 8, (0, 0, 0), -1)
        cv2.putText(warped, f"Arm ({arm_pos[0]}, {arm_pos[1]})", (arm_pos[0]+10, arm_pos[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # === 顯示速度 ===
        cv2.putText(warped, f"Speed: {speed_cmps:.1f} cm/s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Tracking", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()