import pyrealsense2 as rs
import numpy as np
import cv2
import time

# 初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 初始化變量
prev_position = None
prev_time = None
speed = 0
alpha = 0.5  # 高斯平滑因子
position_threshold = 2.3  # 位置改變閾值（像素）

# 將深度座標轉換為實際單位
depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

try:
    while True:
        # 獲取相機幀
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # 將影像轉換為 NumPy 格式
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 使用 OpenCV 檢測曲棍球
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lower_color = np.array([40,50,50])  # 根據曲棍球顏色調整
        upper_color = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 尋找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            current_position = (int(x), int(y))  # 將座標取整數

            # 半徑過濾
            if radius > 5:
                current_time = time.time()

                # 取得 z 軸深度值
                z = int(depth_frame.get_distance(int(x), int(y)) / depth_scale)  # 轉換為整數毫米值

                # 顯示三軸座標
                cv2.putText(color_image, f"Position: ({current_position[0]}, {current_position[1]}, {z})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 更新速度計算（僅對 x, y）
                if prev_position is not None and prev_time is not None:
                    delta_time = current_time - prev_time
                    delta_x = current_position[0] - prev_position[0]
                    delta_y = current_position[1] - prev_position[1]
                    distance = np.sqrt(delta_x**2 + delta_y**2)

                    # 檢查位置改變閾值
                    if distance < position_threshold:
                        speed = 0
                    else:
                        raw_speed = distance / delta_time
                        speed = alpha * raw_speed + (1 - alpha) * speed

                prev_position = current_position
                prev_time = current_time

                # 顯示速度
                cv2.putText(color_image, f"Speed: {speed:.2f} pixels/sec", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 畫出曲棍球位置
                cv2.circle(color_image, current_position, int(radius), (255, 0, 0), 2)
                cv2.circle(color_image, current_position, 5, (0, 255, 0), -1)

        # 顯示結果
        cv2.imshow('Color Image', color_image)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
