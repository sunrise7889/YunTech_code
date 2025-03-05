import cv2
import numpy as np
import pyrealsense2 as rs

# 初始化 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 60)
pipeline.start(config)

# 目標影像解析度
target_width, target_height = 800, 600  

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # 轉換為 NumPy 陣列
    frame = np.asanyarray(color_frame.get_data())

    # 轉換為 HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 設定顏色範圍 (黑色標記)
    lower = np.array([120, 36, 14])
    upper = np.array([10, 255, 255])

    # 過濾特定顏色
    mask = cv2.inRange(hsv, lower, upper)

    # 找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))

    if len(points) == 4:
        # 排序四個點
        points = sorted(points, key=lambda p: (p[0] + p[1]))  # 根據 (x + y) 排序
        top_left, top_right, bottom_left, bottom_right = points

        # 設定目標影像座標 (俯視視角)
        target_points = np.array([[0, 0], [target_width, 0], [0, target_height], [target_width, target_height]], dtype=np.float32)

        # 計算 Homography 變換矩陣
        image_points = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)
        H = cv2.getPerspectiveTransform(image_points, target_points)

        # 透視變換
        warped_image = cv2.warpPerspective(frame, H, (target_width, target_height), flags=cv2.INTER_CUBIC)

        # 繪製邊界
        cv2.polylines(warped_image, [np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("Warped Image (Top-Down View)", warped_image)

    # 顯示原始影像 & 遮罩
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Mask", mask)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
