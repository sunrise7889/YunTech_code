import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)

# 啟動相機
pipeline.start(config)

# 取得相機內部參數
profile = pipeline.get_active_profile()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# 手動標定 4 個角的影像座標 (像素座標)
image_points = np.array([
    [50, 37],  # 左上角
    [1237, 34],  # 右上角
    [1259,652],  # 右下角
    [33,635]   # 左下角
], dtype=np.float32)

# 設定目標平面上的 4 個角點 (俯視視角)
target_width = 500  # 設定校正後影像的寬度
target_height = 300  # 設定校正後影像的高度
target_points = np.array([
    [0, 0],  # 左上角
    [target_width - 1, 0],  # 右上角
    [target_width - 1, target_height - 1],  # 右下角
    [0, target_height - 1]  # 左下角
], dtype=np.float32)

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    
    # 轉換為 numpy 陣列
    color_image = np.asanyarray(color_frame.get_data())
    
    # 計算 Homography 矩陣
    H = cv2.getPerspectiveTransform(image_points, target_points)

    
    # 透視變換影像
    warped_image = cv2.warpPerspective(color_image, H, (target_width, target_height), flags=cv2.INTER_CUBIC)

    
    # 在影像上繪製 4 個角點
    for pt in image_points:
        cv2.circle(color_image, tuple(pt.astype(int)), 5, (0, 0, 255), -1)  # 紅色標記
    cv2.polylines(color_image, [image_points.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # 繪製四邊形邊界
    cv2.polylines(warped_image, [target_points.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)

    # 顯示原始影像 & 轉換後的俯視影像
    cv2.imshow("Warped Image (Top-Down View)", warped_image)
    cv2.imshow("Original Image with Points", color_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
pipeline.stop()
