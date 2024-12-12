import pyrealsense2 as rs
import numpy as np
import cv2

def detect_green_contours(image):
    # 轉換為HSV顏色空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定義綠色的HSV範圍（針對純綠色）
    lower_green = np.array([0, 0, 0])
    upper_green = np.array([179, 40, 255])
    
    # 創建顏色遮罩
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 形態學操作去噪
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, mask

# 設置RealSense管線
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 開始串流
pipeline.start(config)

try:
    while True:
        # 等待新的圖像幀
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        # 轉換為numpy數組
        color_image = np.asanyarray(color_frame.get_data())
        
        # 偵測綠色輪廓
        contours, mask = detect_green_contours(color_image)
        
        # 複製原始影像以繪製輪廓
        image_with_contours = color_image.copy()
        
        # 繪製所有找到的輪廓
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
        
        # 顯示結果
        cv2.imshow('Original Image', color_image)
        cv2.imshow('Green Mask', mask)
        cv2.imshow('Contours', image_with_contours)
        
        # 對每個輪廓進行詳細分析
        for contour in contours:
            # 計算輪廓面積
            area = cv2.contourArea(contour)
            
            # 只分析面積大於一定值的輪廓
            if area > 100:
                # 計算輪廓周長
                perimeter = cv2.arcLength(contour, True)
                
                # 計算最小外接圓
                (x, y), radius = cv2.minEnclosingCircle(contour)

        
        # # 按'q'退出
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()