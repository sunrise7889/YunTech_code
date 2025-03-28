import pyrealsense2 as rs
import numpy as np
import cv2
import time

# 初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 設定 HSV 範圍
lower_yellow = np.array([20, 104, 100], dtype=np.uint8)
upper_yellow = np.array([28, 255, 183], dtype=np.uint8)

lower_green = np.array([40, 50, 50], dtype=np.uint8)
upper_green = np.array([90, 255, 255], dtype=np.uint8)

prev_position = None

# 記錄軌跡與時間
trajectory_yellow = []
trajectory_green = []
time_stamps = []

# 記錄前一幀座標
prev_yellow_pos = None
prev_green_pos = None

print_interval = 0.5  # 每 0.5 秒輸出一次
last_print_time = time.time()
print("等待 3 秒，穩定相機畫面...")
time.sleep(3)

# 設定最大軌跡長度
MAX_TRAJECTORY = 50

# Homography 變換矩陣與選擇的四個點
points = []
H = None
initialized = False  # 是否已設定四個點

# 設定幀數間隔
frame_count = 0
FRAME_SKIP = 5  # 每 3 幀才更新 prev_pos

# 記錄速度歷史
speed_history = []  # 用來存儲歷史速度數據

# 設置鼠標回調函數
def mouse_callback(event, x, y, flags, param):
    global points, H, initialized
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"選擇點: {x}, {y}")
        if len(points) == 4:
            dst_pts = np.array([[0, 0], [639, 0], [639, 479], [0, 479]], dtype=np.float32)
            H, _ = cv2.findHomography(np.array(points, dtype=np.float32), dst_pts)
            initialized = True
            print("Homography 計算完成！")

cv2.namedWindow("RealSense Color")
cv2.setMouseCallback("RealSense Color", mouse_callback)

# 追蹤物件
def track_object(hsv_image, lower_HSV, upper_HSV, trajectory, timestamps):
    """ 追蹤物件並回傳當前 (X, Y) 座標 """
    mask = cv2.inRange(hsv_image, lower_HSV, upper_HSV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            trajectory.append((cX, cY))
            timestamps.append(time.time())
            return (cX, cY), mask
    
    return None, mask


try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 轉換成 NumPy 陣列
        color_image = np.asanyarray(color_frame.get_data())

        # Homography 校正
        if H is not None:
            color_image = cv2.warpPerspective(color_image, H, (640, 480))

        # 轉換 BGR → HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        if initialized:  # 只有設定四個點後才開始追蹤與計算
            # 追蹤黃色握把
            yellow_pos, mask_yellow = track_object(hsv_image, lower_yellow, upper_yellow, trajectory_yellow, time_stamps)
            if yellow_pos:
                cv2.circle(color_image, yellow_pos, 5, (0, 0, 255), -1)
                if len(trajectory_yellow) > 1:
                    for i in range(1, len(trajectory_yellow)):
                        cv2.line(color_image, trajectory_yellow[i - 1], trajectory_yellow[i], (0, 255, 255), 2)

            # 追蹤綠色冰球
            
            green_pos, mask_green = track_object(hsv_image, lower_green, upper_green, trajectory_green, time_stamps)
            contours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if green_pos:
                cv2.circle(color_image, green_pos, 5, (255, 0, 0), -1)
                if len(trajectory_green) > 1:
                    for i in range(1, len(trajectory_green)):
                        cv2.line(color_image, trajectory_green[i - 1], trajectory_green[i], (0, 255, 0), 2)
                # **速度計算**
                c = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(c)
                current_position = (int(x), int(y))
                current_time = time.time()
                if prev_position and prev_time:
                    delta_time = current_time - prev_time
                    delta_x = current_position[0] - prev_position[0]
                    delta_y = current_position[1] - prev_position[1]
                    distance = np.sqrt(delta_x**2 + delta_y**2)

                    if distance < 2.3:
                        speed = 0
                    else:
                        raw_speed = distance / delta_time
                        speed = 0.5 * raw_speed + (1 - 0.5) * speed  # 平滑處理
                prev_position, prev_time = current_position, current_time
                
                # 偵測邊界碰撞
                if green_pos[0] <= 5 or green_pos[0] >= 635 or green_pos[1] <= 5 or green_pos[1] >= 475:
                    print("⚠ 冰球碰到邊界！")

            # **每 FRAME_SKIP 幀才更新 prev_pos**
            if frame_count % FRAME_SKIP == 0:
                prev_yellow_pos = yellow_pos
                prev_green_pos = green_pos
            frame_count += 1

            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                print(f"握把當前座標: {yellow_pos}, 前一幀座標: {prev_yellow_pos}")
                print(f"冰球當前座標: {green_pos}, 前一幀座標: {prev_green_pos}")
                print("-" * 50)
                last_print_time = current_time  # 更新上次 print 時間
            cv2.putText(color_image, f"Speed: {speed:.2f} ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            
        # 限制軌跡數量
        if len(trajectory_yellow) > MAX_TRAJECTORY:
            trajectory_yellow.pop(0)
        if len(trajectory_green) > MAX_TRAJECTORY:
            trajectory_green.pop(0)
            time_stamps.pop(0)

        # 顯示影像
        cv2.imshow("RealSense Color", color_image)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
