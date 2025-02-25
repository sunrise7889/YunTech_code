import cv2
import numpy as np
import pyrealsense2 as rs

# 初始化變數
points_2d = []  # 存放點擊的 2D 座標
points_3d = []  # 存放對應的 3D 座標
frame = None
depth_frame = None

# 初始化 Realsense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
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
        depth = depth_frame.get_distance(x, y)  # 這裡回傳的是 meters

        # 轉換為 3D 座標
        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
        points_3d.append(point_3d)

        print(f"點擊的座標: (x={x}, y={y}), 深度 Z={depth:.3f}m")
        print(f"對應的 3D 座標: {point_3d}")

        # 在畫面上標記
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", frame)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data()).copy()
        cv2.imshow("Select Points", frame)
        cv2.setMouseCallback("Select Points", select_points)

        if len(points_2d) == 4:  # 獲取四個點後退出
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()