import cv2
import numpy as np
import pyrealsense2 as rs
import time

# 初始化變數
points_2d = []
frame = None
depth_frame = None
prev_position = None
prev_time = None
speed = 0

# 初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# 滑鼠點擊回調函數
def select_points(event, x, y, flags, param):
    global points_2d, frame, depth_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        points_2d.append((x, y))  
        print(f"選取點: (x={x}, y={y})")

        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", frame)

# **開始點擊四個角來選取桌面**
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    frame = np.asanyarray(color_frame.get_data()).copy()

    cv2.imshow("Select Points", frame)
    cv2.setMouseCallback("Select Points", select_points)

    if len(points_2d) == 4:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 設定俯視圖的尺寸
target_width, target_height = 600, 300  
target_points = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)

# 計算透視變換矩陣
image_points = np.array(points_2d, dtype=np.float32)
H = cv2.getPerspectiveTransform(image_points, target_points)
inv_H = np.linalg.inv(H)

# 定義桌面邊界
edges = [
    (target_points[0], target_points[1]),  # 上邊
    (target_points[1], target_points[2]),  # 右邊
    (target_points[2], target_points[3]),  # 下邊
    (target_points[3], target_points[0])   # 左邊
]

# **開始冰球追蹤與碰撞檢測**
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    warped_image = cv2.warpPerspective(color_image, H, (target_width, target_height), flags=cv2.INTER_LINEAR)

    # **顏色範圍檢測冰球 (HSV)**
    hsv = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([40, 50, 50])  
    upper_color = np.array([80, 255, 255])  
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius > 5:
            current_position = (int(x), int(y))
            current_time = time.time()

            # **轉換回原始影像座標**
            orig_pos = np.dot(inv_H, np.array([x, y, 1]))
            orig_x, orig_y = orig_pos[:2] / orig_pos[2]
            orig_x, orig_y = int(orig_x), int(orig_y)

            # **獲取深度值**
            z = int(depth_frame.get_distance(orig_x, orig_y) / depth_scale)

            # **速度計算**
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

            # **檢測碰撞**
            collision = False
            for (p1, p2) in edges:
                # 計算點到線的距離
                x0, y0 = current_position
                x1, y1 = p1
                x2, y2 = p2

                # 計算線段長度
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if line_length == 0:
                    continue  # 避免除以 0

                # 計算垂直距離
                distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / line_length

                if distance < radius:  # 當冰球中心與邊界距離小於半徑，視為碰撞
                    collision = True
                    print(f"Collision detected at (x={x0}, y={y0})")
                    cv2.circle(warped_image, (x0, y0), int(radius), (0, 0, 255), 3)  # 標記紅色碰撞區域

            # **畫出冰球位置與速度**
            cv2.putText(warped_image, f"Speed: {speed:.2f} ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.circle(warped_image, current_position, int(radius), (255, 0, 0), 2)

    # **畫出桌面邊界**
    for (p1, p2) in edges:
        cv2.line(warped_image, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 2)

    cv2.imshow("Warped Image (Top-Down View)", warped_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
pipeline.stop()
