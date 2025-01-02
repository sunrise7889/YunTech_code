import cv2
import numpy as np
import pyrealsense2 as rs

def detect_green_circles(color_frame, depth_frame):
    # 將影像轉換為numpy數組
    frame = np.asanyarray(color_frame.get_data())
    
    # 將影像轉換到HSV色彩空間
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 定義綠色的HSV範圍（可能需要根據實際環境調整）
    lower_green = np.array([40,50,50])
    upper_green = np.array([80,255,255])
    
    # 建立遮罩，只保留綠色區域
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 形態學操作，去除雜訊
    kernel = np.ones((5,5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    # 轉換為灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 在綠色遮罩上進行圓形檢測
    green_gray = cv2.bitwise_and(gray, gray, mask=green_mask)
    
    # 高斯模糊，減少雜訊
    blurred = cv2.GaussianBlur(green_gray, (9, 9), 2)
    
    # 使用霍夫圓形檢測
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT,
        dp=1,              # 解析度縮放因子
        minDist=50,        # 圓心之間的最小距離
        param1=45,         # Canny邊緣檢測的高閾值
        param2=28,         # 圓形檢測的閾值
        minRadius=10,      # 最小圓半徑
        maxRadius=200      # 最大圓半徑
    )
    
    return circles, green_mask, depth_frame

def draw_coordinate_box(image, text, position, font_scale=0.7):
    """
    在圖像上繪製帶有背景框的座標文字
    """
    # 獲取文字大小
    (text_width, text_height), baseline = cv2.getTextSize(
        text, 
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, 
        2
    )
    
    # 計算背景框的位置
    x, y = position
    padding = 5
    
    # 繪製背景框（黑色半透明背景）
    cv2.rectangle(
        image,
        (x, y - text_height - padding),
        (x + text_width + padding * 2, y + padding),
        (0, 0, 0),
        -1
    )
    
    # 繪製文字（白色）
    cv2.putText(
        image,
        text,
        (x + padding, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2
    )

def main():
    # 設置RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 啟用彩色和深度流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 啟動攝影機
    pipeline.start(config)
    
    # 建立對齊物件
    align = rs.align(rs.stream.color)
    
    try:
        while True:
            # 等待幀
            frames = pipeline.wait_for_frames()
            
            # 對齊彩色和深度幀
            aligned_frames = align.process(frames)
            
            # 取得彩色和深度幀
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # 偵測綠色圓形             
            circles, green_mask, depth_frame = detect_green_circles(color_frame, depth_frame)
            
            # 創建一個彩色版本的綠色遮罩，用於繪製
            green_mask_color = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
            
            # 如果檢測到圓形
            if circles is not None:
                # 將圓形轉換為整數座標
                circles = np.uint16(np.around(circles))
                
                # 繪製圓形
                for i in circles[0, :]:
                    # 取得深度值（單位：毫米）
                    depth = depth_frame.get_distance(i[0], i[1])
                    
                    # 在綠色遮罩上繪製圓周（粗線條）
                    cv2.circle(green_mask_color, (i[0], i[1]), i[2], (0, 255, 0), 3)
                    
                    # 繪製明顯的圓心（較大半徑）
                    cv2.circle(green_mask_color, (i[0], i[1]), 4, (0, 0, 255), -1)
                    
                    # 準備座標文字
                    coord_text = f"X:{i[0]} Y:{i[1]} Z:{depth*1000:.0f}mm"
                    
                    #顯示座標
                    print(coord_text)
                    
                    
                    # 使用新的繪製函數來顯示座標
                    draw_coordinate_box(
                        green_mask_color,
                        coord_text,
                        (i[0] + 10, i[1] + 30)
                    )
            
            # 顯示綠色遮罩
            cv2.imshow('Green Mask with Circles', green_mask_color)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # 關閉管線
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()