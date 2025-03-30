import pyrealsense2 as rs
import numpy as np
import cv2
import time

#  初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

#  設定 HSV 範圍
lower_yellow = np.array([20, 104, 100], dtype=np.uint8)
upper_yellow = np.array([28, 255, 183], dtype=np.uint8)

lower_green = np.array([40, 50, 50], dtype=np.uint8)
upper_green = np.array([90, 255, 255], dtype=np.uint8)

#  記錄軌跡
trajectory_yellow = []
trajectory_green = []

#  記錄前一幀座標
prev_yellow_pos = None
prev_green_pos = None


print_interval = 0.5  # 每 0.5 秒輸出一次
last_print_time = time.time()
print("等待 3 秒，穩定相機畫面...")
time.sleep(3)

def track_object(hsv_image, lower_HSV, upper_HSV, trajectory):
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
            return (cX, cY), mask
    
    print("⚠ 未偵測到物件")  #  確認是否真的沒抓到
    return None, mask

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        #  轉換成 NumPy 陣列
        color_image = np.asanyarray(color_frame.get_data())

        #  轉換 BGR → HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        #  追蹤黃色握把
        yellow_pos, mask_yellow = track_object(hsv_image, lower_yellow, upper_yellow, trajectory_yellow)
        if yellow_pos:
            cv2.circle(color_image, yellow_pos, 5, (0, 0, 255), -1)  # 標記當前握把位置
            if len(trajectory_yellow) > 1:
                for i in range(1, len(trajectory_yellow)):
                    cv2.line(color_image, trajectory_yellow[i - 1], trajectory_yellow[i], (0, 255, 255), 2)
                    prev_yellow_pos = yellow_pos

        # 追蹤綠色冰球
        green_pos, mask_green = track_object(hsv_image, lower_green, upper_green, trajectory_green)
        if green_pos:
            cv2.circle(color_image, green_pos, 5, (255, 0, 0), -1)  # 標記當前冰球位置
            if len(trajectory_green) > 1:
                for i in range(1, len(trajectory_green)):
                    cv2.line(color_image, trajectory_green[i - 1], trajectory_green[i], (0, 255, 0), 2)
                    prev_green_pos = green_pos

        MAX_TRAJECTORY = 50  # 設定最多存 50 個點
        if len(trajectory_yellow) > MAX_TRAJECTORY:
            trajectory_yellow.pop(0)  # 移除最舊的點
        if len(trajectory_green) > MAX_TRAJECTORY:
            trajectory_green.pop(0)
        
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            print(f"握把當前座標: {yellow_pos}, 前一幀座標: {prev_yellow_pos}")
            print(f"冰球當前座標: {green_pos}, 前一幀座標: {prev_green_pos}")
            print("-" * 50)
            last_print_time = current_time  # 更新上次 print 時間
        
            
        #  顯示影像
        cv2.imshow("RealSense Color", color_image)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
