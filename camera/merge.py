import cv2
import numpy as np
import pyrealsense2 as rs
import time


#點擊順序:從左上開始順時針繞


prev_position = None
prev_time = None
speed = 0

# 初始化變數
points_2d = []  # 存放點擊的 2D 座標
points_3d = []  # 存放對應的 3D 座標
frame = None
depth_frame = None

# 初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)

# 取得深度感測器的 scale，將深度值轉換為真實世界單位 (米)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# 取得相機內參
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# 滑鼠點擊回調函數
def select_points(event, x, y, flags, param):
    global points_2d, points_3d, frame, depth_frame, depth_scale

    if event == cv2.EVENT_LBUTTONDOWN:
        points_2d.append((x, y))  # 儲存 2D 座標
        
        # 取得深度值 (mm) 並轉換為米 (m)
        depth = depth_frame.get_distance(x, y)  # 這裡回傳的是米

        # 轉換為 3D 座標
        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
        points_3d.append(point_3d)

        print(f"點擊的座標: (x={x}, y={y}), 深度 Z={depth:.3f}m")
        print(f"對應的 3D 座標: {point_3d}")

        # 在畫面上標記
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", frame)

# 透視變換處理
target_width, target_height = 500, 300
target_points = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    frame = np.asanyarray(color_frame.get_data()).copy()

    # 顯示畫面並等待使用者選取四個點
    cv2.imshow("Select Points", frame)
    cv2.setMouseCallback("Select Points", select_points)

    # 確保選取四個點
    if len(points_2d) == 4:  # 獲取四個點後退出
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 取得透視變換矩陣
image_points = np.array(points_2d, dtype=np.float32)
H = cv2.getPerspectiveTransform(image_points, target_points)  # 透視變換矩陣
inv_H = np.linalg.inv(H)  # 逆變換矩陣

# 處理影像並計算速度
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    warped_image = cv2.warpPerspective(color_image, H, (target_width, target_height), flags=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([40, 50, 50])  # 綠色區間
    upper_color = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    cv2.imshow("HSV Mask", mask)  # Debug: 確保遮罩正確

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
                    speed = 0.5 * raw_speed + (1 - 0.5) * speed

            prev_position, prev_time = current_position, current_time

            cv2.putText(warped_image, f"Speed: {speed:.2f} pixels/sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.circle(warped_image, current_position, int(radius), (255, 0, 0), 2)

    cv2.polylines(warped_image, [target_points.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imshow("Warped Image (Top-Down View)", warped_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
pipeline.stop()
