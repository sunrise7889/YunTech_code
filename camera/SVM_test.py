import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

def detect_arm_position(color_image):
    """ 偵測手臂的位置，回傳 (X, Y) 座標 """
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    
    # 定義膚色範圍 (可根據環境調整)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # 找出輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
    return None

prev_arm_position = None  # 前一幀手臂座標

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        color_image = np.asanyarray(color_frame.get_data())
        arm_position = detect_arm_position(color_image)
        
        if arm_position:
            print(f"當前手臂座標: {arm_position}")
            if prev_arm_position:
                print(f"前一幀手臂座標: {prev_arm_position}")
            prev_arm_position = arm_position  # 更新前一幀座標
        
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
