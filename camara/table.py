import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 初始化 RealSense 設備
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 啟動相機
    pipeline.start(config)
    
    # 設定觸碰偵測閾值
    depth_threshold = 500  # 根據場景調整閾值 (單位: mm)
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # 轉換影像格式
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # 定義桌上型冰球桌的邊界框 (手動設定範圍)
            mask = np.zeros_like(gray_image)
            points = np.array([[50, 50], [590, 50], [590, 430], [50, 430]])  # 調整邊界
            cv2.fillPoly(mask, [points], 255)
            
            # 套用遮罩
            masked_image = cv2.bitwise_and(color_image, color_image, mask=mask)
            
            # 繪製中線
            cv2.line(masked_image, (320, 50), (320, 430), (0, 0, 255), 2)
            
            # 偵測圓形物體 (霍夫變換)
            circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                       param1=50, param2=30, minRadius=10, maxRadius=50)
            
            object_detected = False
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    x, y, r = circle
                    
                    # 檢查是否在邊界內 (簡單判斷圓心是否接近邊界)
                    if x - r <= 50 or x + r >= 590 or y - r <= 50 or y + r >= 430:
                        object_detected = True
                        cv2.circle(masked_image, (x, y), r, (0, 255, 0), 2)  # 標記偵測到的圓
                        cv2.putText(masked_image, "Object Detected!", (200, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(masked_image, (50, 50), (590, 430), (0, 0, 255), 3)
            
            # 顯示結果
            cv2.imshow('Detected Table', masked_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
