import pyrealsense2 as rs
import numpy as np
import cv2

# ✅ 初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 初始 HSV 範圍 (點擊後更新)
lower_HSV = np.array([0, 50, 50], dtype=np.uint8)
upper_HSV = np.array([20, 255, 255], dtype=np.uint8)

def pick_HSV(event, x, y, flags, param):
    """ 點擊畫面時，選擇 HSV 顏色範圍 """
    global lower_HSV, upper_HSV, hsv_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv_image[y, x]  # 取得該點的 HSV 值
        
        # 設定 HSV 範圍
        h, s, v = pixel
        lower_HSV = np.array([max(h - 10, 0), max(s - 40, 0), max(v - 40, 0)], dtype=np.uint8)
        upper_HSV = np.array([min(h + 10, 179), min(s + 40, 255), min(v + 40, 255)], dtype=np.uint8)
        
        print(f"選取 HSV: {h}, {s}, {v}")
        print(f"設定範圍: {lower_HSV} ~ {upper_HSV}")

# ✅ 設定 OpenCV 視窗與點擊事件
cv2.namedWindow("RealSense Color")
cv2.setMouseCallback("RealSense Color", pick_HSV)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # ✅ 轉換成 NumPy 陣列
        color_image = np.asanyarray(color_frame.get_data())

        # ✅ 轉換 BGR → HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # ✅ 產生 HSV 遮罩
        mask = cv2.inRange(hsv_image, lower_HSV, upper_HSV)

        # ✅ 顯示影像
        cv2.imshow("RealSense Color", color_image)  # 原始影像
        cv2.imshow("Skin Mask", mask)  # 遮罩結果

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
