import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 固定黃色 HSV 範圍 (適用於冰球桌握把)
lower_HSV = np.array([20, 104, 100], dtype=np.uint8)
upper_HSV = np.array([28, 255, 183], dtype=np.uint8)

# 用來記錄路徑的陣列
trajectory = []

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 轉換成 NumPy 陣列
        color_image = np.asanyarray(color_frame.get_data())

        # 轉換 BGR → HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # 產生 HSV 遮罩
        mask = cv2.inRange(hsv_image, lower_HSV, upper_HSV)

        # 找出黃色物件的中心
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 記錄路線
                trajectory.append((cX, cY))

                # 在畫面上標記當前位置
                cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)

        # 在畫面上畫出軌跡
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                cv2.line(color_image, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

        # 顯示影像
        cv2.imshow("RealSense Color", color_image)  # 原始影像
        cv2.imshow("Mask", mask)  # 遮罩結果

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
