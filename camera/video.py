import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 建立輸出資料夾
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# 初始化 RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# 啟動 RealSense
pipeline.start(config)

frame_count = 0  # 計算總幀數
save_interval = 30  # 設定每幾幀存一次

try:
    while True:
        # 取得影像幀
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 轉換成 NumPy 陣列
        color_image = np.asanyarray(color_frame.get_data())

        # 每 N 幀存一次圖片
        if frame_count % save_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, color_image)
            print(f"Saved: {frame_filename}")

        frame_count += 1

        # 顯示影像
        cv2.imshow("RGB Frame", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 鍵退出
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
