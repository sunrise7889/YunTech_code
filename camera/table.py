import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 初始化 RealSense 設備
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720 ,rs.format.bgr8, 30)
    
    # 啟動相機
    pipeline.start(config)
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # 轉換影像格式
            color_image = np.asanyarray(color_frame.get_data())

            # 轉換到 HSV 色彩空間
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            # 定義綠色範圍
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])

            # 建立遮罩，只保留綠色區域
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            # 偵測綠色圓形物體 (霍夫變換)
            circles = cv2.HoughCircles(mask_green, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                       param1=50, param2=30, minRadius=10, maxRadius=50)

            # 定義桌上型冰球桌的邊界框 (正確範圍)
            table_x1, table_y1 = 10, 10   # 左上角
            table_x2, table_y2 = 1270, 710 # 右下角
            cv2.rectangle(color_image, (table_x1, table_y1), (table_x2, table_y2), (255, 0, 0), 3)

            # 繪製中線
            cv2.line(color_image, (640, 10), (640, 710), (0, 0, 255), 2)

            object_detected = False
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    x, y, r = circle
                    
                    # 檢查圓形是否碰觸邊界
                    if x - r <= table_x1 or x + r >= table_x2 or y - r <= table_y1 or y + r >= table_y2:
                        object_detected = True
                        cv2.circle(color_image, (x, y), r, (0, 255, 0), 2)  # 用綠色標記圓
                        cv2.putText(color_image, "Green Object Detected!", (200, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 縮小顯示畫面
            resized_image = cv2.resize(color_image, (800, 450))  # 縮小到 800x450
            cv2.imshow('Detected Table', resized_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
