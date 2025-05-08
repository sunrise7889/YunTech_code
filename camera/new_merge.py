import cv2
import numpy as np
import pyrealsense2 as rs
import time
import joblib
import pandas as pd
from collections import deque

# === 變數定義 ===
pred = joblib.load("svm_model_no_scaler.pkl")

selected_roi = None  # 新增初始化
x0, y0 = 0, 0  # 滑鼠選取起始點
drawing = False
template = None
defense_roi = None
# 參數設定
arm_pos = [300, 150]  # 手臂位置模擬
intersection_points = []  # 交點儲存
grid_cols = 6  # 橫向幾格（x方向）
grid_rows = 16  # 縱向幾格（y方向）
SVMlock = 0
PUCK_RADIUS = 17  # 冰球半徑 (px)
target_width, target_height = 600, 300
px_to_cm = 120 / target_width  # 桌面寬度 120cm 對應 600px
PRINT_INTERVAL = 0.3
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
last_print_time = time.time()

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

# 其他變數
defense_roi = None
center_line_x = target_width // 2
right_top_half = top_bound
right_bottom_half = (top_bound + bottom_bound) // 2

#高低速區參數
SPEED_THRESHOLD = 30  # cm/s
EARLY_OFFSET_CM = 20  # 提前擊球距離（低速用）
EARLY_OFFSET_PX = EARLY_OFFSET_CM / px_to_cm

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

def is_ball_in_defense(cx, cy):
    """檢查球是否接觸防守區域（考慮半徑）"""
    if defense_roi is None:
        return False
    defense_left, defense_right, defense_top, defense_bottom = defense_roi
    return (cx - PUCK_RADIUS < defense_right and 
            cx + PUCK_RADIUS > defense_left and 
            cy - PUCK_RADIUS < defense_bottom and 
            cy + PUCK_RADIUS > defense_top)

def predict_trajectory(start_pos, velocity, max_bounces=3):
    """預測完整軌跡（考慮多次反射），並在球越過防守線後停止預測"""
    vx, vy = velocity
    
    # 如果球往左移動（vx <= 0），不預測
    if vx <= 0:
        return []  # 返回空列表，不畫軌跡
    
    trajectory = []
    current_pos = np.array(start_pos)
    current_vel = np.array(velocity)
    
    for _ in range(max_bounces + 1):
        # 檢查是否已經越過防守線
        if current_pos[0] > defense_center_x:
            break  # 停止預測
            
        # 預測下一個碰撞點
        collision_pos, boundary = predict_collision_with_radius(
            current_pos[0], current_pos[1], current_vel[0], current_vel[1])
        
        if not collision_pos:
            break  # 沒有碰撞點（出界）
            
        # 如果碰撞點在防守線右側，則截斷到防守線
        if collision_pos[0] > defense_center_x:
            # 計算到達防守線的時間
            t = (defense_center_x - current_pos[0]) / current_vel[0]
            collision_pos = (
                defense_center_x,
                current_pos[1] + current_vel[1] * t
            )
            trajectory.append((tuple(current_pos), tuple(collision_pos)))
            break  # 停止進一步預測
            
        trajectory.append((tuple(current_pos), tuple(collision_pos)))
        
        # 計算反射後向量
        current_vel = np.array(calculate_reflection(current_vel[0], current_vel[1], boundary))
        current_pos = np.array(collision_pos)
    
    return trajectory

# === 滑鼠事件：選角點 ===
def select_corners(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_2d) < 4:
        points_2d.append((x, y))
        print(f"點選角點: ({x}, {y})")

def get_grid_intersection(col, row):
    x = defense_left + col * cell_w
    y = defense_top + row * cell_h
    return [int(x), int(y)]

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

# == 自訂防守區域 ==
def defense_callback(event, x, y, flags, param):
    global defense_roi, selecting_defense, x0_d, y0_d, defense_center_x
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting_defense = True
        x0_d, y0_d = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        selecting_defense = False
        x1_d, y1_d = x, y
        defense_roi = (
            min(x0_d, x1_d),
            max(x0_d, x1_d),
            min(y0_d, y1_d),
            max(y0_d, y1_d)
        )
        defense_left, defense_right, defense_top, defense_bottom = defense_roi
        defense_center_x = (defense_left + defense_right) // 2  # 計算防守區域的中間x座標
        print(f"防守區域選取完成：{defense_roi}")


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

# === 防守區域選取 ===
cv2.namedWindow("Select Defense Area")
cv2.setMouseCallback("Select Defense Area", defense_callback)
print("請用滑鼠在俯視影像上框選防守區域...")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    frame = np.asanyarray(color_frame.get_data())
    warped = cv2.warpPerspective(frame, H, (target_width, target_height))
    show = warped.copy()

    if defense_roi:
        l, r, t, b = defense_roi
        cv2.rectangle(show, (l, t), (r, b), (255, 0, 255), 2)
        cv2.imshow("Select Defense Area", show)
        cv2.waitKey(1000)  # 停 1 秒給使用者看選取結果
        break  # 自動跳出迴圈

    cv2.imshow("Select Defense Area", show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # 給使用者保留 q 強制跳出

# 選取完成後關閉視窗並計算中線
cv2.destroyWindow("Select Defense Area")  # 關閉選取視窗
defense_left, defense_right, defense_top, defense_bottom = defense_roi
defense_center_x = (defense_left + defense_right) // 2  # 計算防守線中點



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

    if selected_roi is not None:  # 修改後的判斷條件
        x, y, w, h = selected_roi
        cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 255), 2)
        template = warped[y:y+h, x:x+w].copy()
        print("握把選取完成，開始追蹤...")
        break

    cv2.imshow("Select Handle", show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

cv2.destroyWindow("Select Handle")
vx, vy = 0, 0
# === 第三階段: 開始追蹤 ===
cv2.namedWindow("Tracking")

try:
    while True:
        frames = pipeline.wait_for_frames()
        frame = np.asanyarray(frames.get_color_frame().get_data())
        warped = cv2.warpPerspective(frame, H, (target_width, target_height))
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

        # === 顯示邊界 ===
        cv2.rectangle(warped, (left_bound, top_bound), (right_bound, bottom_bound), (0, 255, 255), 2)
        cv2.line(warped, (right_bound, right_bottom_half), (right_bound, bottom_bound), (0, 0, 255), 2)
        cv2.line(warped, (center_line_x, 0), (center_line_x, target_height), (0, 0, 255), 2)  # 中線

        
        # === 冰球追蹤 ===
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
                
                            
                # === 軌跡預測（只在球往右移動時計算） ===
                if len(prev_history) >= 5 and (abs(dx) > 0.5 or abs(dy) > 0.5):
                    prev_cx_g, prev_cy_g = prev_history[-5]
                    vx = cx_g - prev_cx_g
                    vy = cy_g - prev_cy_g
                    
                    # 只在 vx > 0（球往右移動）時預測
                    if vx > 0:
                        trajectory = predict_trajectory((cx_g, cy_g), (vx, vy))
                        # === 新增：依據球速分流擊球策略 ===


                        if trajectory:
                            last_segment = trajectory[-1]
                            _, predicted_hit_point = last_segment
                            predicted_hit_point = np.array(predicted_hit_point)

                            if speed_cmps > SPEED_THRESHOLD:
                                # 高速：提前到落點等待
                                dist_cm = np.linalg.norm(predicted_hit_point - np.array([cx_g, cy_g])) * px_to_cm
                                time_to_reach = dist_cm / speed_cmps if speed_cmps > 0 else 0
                                print(f"[高速擊球] 預測落點: {predicted_hit_point}, 距離: {dist_cm:.1f}cm, 時間: {time_to_reach:.2f}s")
                                arm_pos = tuple(predicted_hit_point.astype(int))
                                cv2.circle(warped, arm_pos, 8, (0, 255, 0), -1)
                                cv2.line(warped, (int(cx_g), int(cy_g)), arm_pos, (0, 255, 0), 2)
                            else:
                                # 低速：進入防守區就擊球
                                if is_ball_in_defense(cx_g, cy_g):
                                    hit_point = min(intersection_points, key=lambda pt: np.linalg.norm(predicted_hit_point - np.array(pt)))
                                    print(f"[低速擊球] 球進入防守區，移動至交點: {hit_point}")
                                    arm_pos = hit_point
                                    cv2.circle(warped, hit_point, 8, (0, 165, 255), -1)
                                    cv2.line(warped, (int(cx_g), int(cy_g)), hit_point, (0, 165, 255), 2)
                        # 繪製預測軌跡
                        for i, (start, end) in enumerate(trajectory):
                            color = (0, 255, 255) if i == 0 else (0, 0, 255)  # 第一段黃色，後續紅色
                            cv2.line(warped, (int(start[0]), int(start[1])), 
                                    (int(end[0]), int(end[1])), color, 2)
                prev_g_pos = (cx_g, cy_g)
                prev_time = now
        
        # === 握把追蹤 ===
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
                else:
                    prev_handle = None           
        
        # === 畫防守格子 ===
        if defense_roi:
            defense_left, defense_right, defense_top, defense_bottom = defense_roi
            cell_w = (defense_right - defense_left) // grid_cols
            cell_h = (defense_bottom - defense_top) // grid_rows

            for row in range(grid_rows):
                for col in range(grid_cols):
                    x1 = defense_left + col * cell_w
                    y1 = defense_top + row * cell_h
                    x2 = x1 + cell_w
                    y2 = y1 + cell_h
                    #cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    grid_id = row * grid_cols + col
            # === 新增：畫防守區域的中間垂直線（防守線） ===
            cv2.line(warped, 
                    (defense_center_x, defense_top),  # 起點 (x, y_top)
                    (defense_center_x, defense_bottom),  # 終點 (x, y_bottom)
                    (0, 0, 255),  # 顏色 (BGR格式，這裡是洪色)
                    2)  # 線條粗細   
      
            # 額外畫交點（新增）
            intersection_points = []  # 放在區塊外初始化也可以
            for row in range(grid_rows + 1):  # 多一行交點
                for col in range(grid_cols + 1):  # 多一列交點
                    x = defense_left + col * cell_w
                    y = defense_top + row * cell_h
                    intersection_points.append((x, y))
                    cv2.circle(warped, (x, y), 3, (0, 255, 255), -1)  # 黃色小點交點
        
        # == 模擬手臂 ===
        cv2.circle(warped, tuple(arm_pos), 8, (0, 0, 0), -1)
        cv2.putText(warped, f"Arm ({arm_pos[0]}, {arm_pos[1]})", (arm_pos[0]+10, arm_pos[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 
        
        if cx_g is not None and cy_g is not None:
            puck_pos = np.array([cx_g, cy_g])
            closest = min(intersection_points, key=lambda pt: np.linalg.norm(puck_pos - np.array(pt)))  
            # 如果球正在接近防守區域（距離防守區域左邊界 < 一定值）
            defense_dist = defense_left - cx_g
            if defense_dist > 0 and defense_dist < 150:  # 調整 150px 為觸發距離
                arm_pos = closest  # 移動手臂到最近的交點
                cv2.circle(warped, closest, 6, (0, 0, 255), 2)  # 標記最近交點

            
        # === 手臂移動（只在球往右移動時觸發） ===
        if prev_handle is not None and vx > 0:  # 新增 vx > 0 條件
            prev_x, prev_y = prev_handle
            if (cx_h > prev_x) and (abs(cx_h - prev_x) > 10) and (SVMlock == 0):
                input_data = pd.DataFrame([[cx_h, cy_h, prev_x, prev_y, cx_g, cy_g]],
                    columns=["hand_x", "hand_y", "prev_hand_x", "prev_hand_y", "ball_x", "ball_y"])
                prediction = pred.predict(input_data)
                
                # 根據預測結果移動到相應的格子
                if prediction == 0:
                    target_col, target_row = 5, 4  # 上半部
                else:
                    target_col, target_row = 5, 12  # 下半部
                    
                arm_pos = get_grid_intersection(target_col, target_row)
                print(f"Prediction: {prediction[0]}, Moving to ({target_col}, {target_row})")
                SVMlock = 1  # 避免重複預測
                
            # 當握把回到左側時重置鎖定
            if cx_h < 100:
                SVMlock = 0
                
            # 如果球接近防守區域，優先移動到最近交點（仍然只在 vx > 0 時觸發）
            defense_dist = defense_left - cx_g
            if cx_g is not None and defense_dist > 0 and defense_dist < 150 and vx > 0:
                closest = min(intersection_points, key=lambda pt: np.linalg.norm(np.array([cx_g, cy_g]) - np.array(pt)))
                arm_pos = closest

        # === 顯示防守區域 ===
        if defense_roi:
            cv2.rectangle(warped, (defense_left, defense_top), (defense_right, defense_bottom), (255, 0, 255), 2)
        
        # === 顯示當前球移動方向 ===
        if 'vx' in locals() and 'vy' in locals():  # 確保變數存在
            if vx > 0.5:  # 加入閾值避免微小晃動
                direction = "→ Right"
                direction_color = (0, 255, 0)  # 綠色
            elif vx < -0.5:
                direction = "← Left" 
                direction_color = (0, 0, 255)  # 紅色
            else:
                direction = "No movement"
                direction_color = (255, 255, 255)  # 白色

            
        cv2.imshow("Tracking", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()