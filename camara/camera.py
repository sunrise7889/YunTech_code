# import pyrealsense2 as rs
# import numpy as np
# import cv2

# def detect_green_contours(image):
#     # 轉換為HSV顏色空間
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # 定義綠色的HSV範圍（針對純綠色）
#     lower_green = np.array([0, 0, 0])
#     upper_green = np.array([179, 40, 255])
    
#     # 創建顏色遮罩
#     mask = cv2.inRange(hsv, lower_green, upper_green)
    
#     # 形態學操作去噪
#     kernel = np.ones((5,5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
#     # 尋找輪廓
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     return contours, mask

# # 設置RealSense管線
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # 開始串流
# pipeline.start(config)

# try:
#     while True:
#         # 等待新的圖像幀
#         frames = pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()
        
#         # 轉換為numpy數組
#         color_image = np.asanyarray(color_frame.get_data())
        
#         # 偵測綠色輪廓
#         contours, mask = detect_green_contours(color_image)
        
#         # 複製原始影像以繪製輪廓
#         image_with_contours = color_image.copy()
        
#         # 繪製所有找到的輪廓
#         cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
        
#         # 顯示結果
#         cv2.imshow('Original Image', color_image)
#         cv2.imshow('Green Mask', mask)
#         cv2.imshow('Contours', image_with_contours)
        
#         # 對每個輪廓進行詳細分析
#         for contour in contours:
#             # 計算輪廓面積
#             area = cv2.contourArea(contour)
            
#             # 只分析面積大於一定值的輪廓
#             if area > 100:
#                 # 計算輪廓周長
#                 perimeter = cv2.arcLength(contour, True)
                
#                 # 計算最小外接圓
#                 (x, y), radius = cv2.minEnclosingCircle(contour)

        
#         # 按'q'退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     pipeline.stop()
#     cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import pyrealsense2 as rs

# def detect_circles(color_frame, depth_frame):
#     # 轉換為灰度圖
#     gray = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2GRAY)
    
#     # 高斯模糊，減少雜訊
#     blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
#     # 使用霍夫圓形檢測
#     circles = cv2.HoughCircles(
#         blurred, 
#         cv2.HOUGH_GRADIENT, 
#         dp=1,               # 解析度縮放因子
#         minDist=50,         # 圓心之間的最小距離
#         param1=50,          # Canny邊緣檢測的高閾值
#         param2=30,          # 圓形檢測的閾值
#         minRadius=10,       # 最小圓半徑
#         maxRadius=200       # 最大圓半徑
#     )
    
#     return circles

# def main():
#     # 設置RealSense
#     pipeline = rs.pipeline()
#     config = rs.config()
    
#     # 啟用彩色和深度流
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
#     # 啟動攝影機
#     pipeline.start(config)
    
#     # 建立對齊物件
#     align = rs.align(rs.stream.color)
    
#     try:
#         while True:
#             # 等待幀
#             frames = pipeline.wait_for_frames()
            
#             # 對齊彩色和深度幀
#             aligned_frames = align.process(frames)
            
#             # 取得彩色和深度幀
#             color_frame = aligned_frames.get_color_frame()
#             depth_frame = aligned_frames.get_depth_frame()
            
#             if not color_frame or not depth_frame:
#                 continue
            
#             # 將幀轉換為numpy數組
#             frame = np.asanyarray(color_frame.get_data())
            
#             # 偵測圓形
#             circles = detect_circles(color_frame, depth_frame)
            
#             # 如果檢測到圓形
#             if circles is not None:
#                 # 將圓形轉換為整數座標
#                 circles = np.uint16(np.around(circles))
                
#                 # 繪製圓形
#                 for i in circles[0, :]:
#                     # 取得深度值（單位：毫米）
#                     depth = depth_frame.get_distance(i[0], i[1])
                    
#                     # 繪製圓周
#                     cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    
#                     # 繪製圓心
#                     cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
                    
#                     # 標註座標（X, Y, Z）
#                     coord_text = f"({i[0]},{i[1]},{depth*1000:.0f}mm)"
#                     cv2.putText(frame, coord_text, 
#                                 (i[0]+10, i[1]+10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
#                                 (255, 255, 255), 1)
            
#             # 顯示影像
#             cv2.imshow('Circle Detection with Depth', frame)
            
#             # 按 'q' 退出
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
    
#     finally:
#         # 關閉管線
#         pipeline.stop()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
# import cv2
# import numpy as np
# import pyrealsense2 as rs

# def detect_green_circles(color_frame, depth_frame):
#     # 將影像轉換為numpy數組
#     frame = np.asanyarray(color_frame.get_data())
    
#     # 將影像轉換到HSV色彩空間
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     # 定義綠色的HSV範圍（可能需要根據實際環境調整）
#     lower_green = np.array([40,50,50])
#     upper_green = np.array([80,255,255])
    
#     # 建立遮罩，只保留綠色區域
#     green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
#     # 形態學操作，去除雜訊
#     kernel = np.ones((5,5), np.uint8)
#     green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
#     green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
#     # 轉換為灰度
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # 在綠色遮罩上進行圓形檢測
#     green_gray = cv2.bitwise_and(gray, gray, mask=green_mask)
    
#     # 高斯模糊，減少雜訊
#     blurred = cv2.GaussianBlur(green_gray, (9, 9), 2)
    
#     # 使用霍夫圓形檢測
#     circles = cv2.HoughCircles(
#         blurred, 
#         cv2.HOUGH_GRADIENT, 
#         dp=1,               # 解析度縮放因子
#         minDist=50,         # 圓心之間的最小距離
#         param1=50,          # Canny邊緣檢測的高閾值
#         param2=30,          # 圓形檢測的閾值
#         minRadius=10,       # 最小圓半徑
#         maxRadius=200       # 最大圓半徑
#     )
    
#     return circles, green_mask

# def main():
#     # 設置RealSense
#     pipeline = rs.pipeline()
#     config = rs.config()
    
#     # 啟用彩色和深度流
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
#     # 啟動攝影機
#     pipeline.start(config)
    
#     # 建立對齊物件
#     align = rs.align(rs.stream.color)
    
#     try:
#         while True:
#             # 等待幀
#             frames = pipeline.wait_for_frames()
            
#             # 對齊彩色和深度幀
#             aligned_frames = align.process(frames)
            
#             # 取得彩色和深度幀
#             color_frame = aligned_frames.get_color_frame()
#             depth_frame = aligned_frames.get_depth_frame()
            
#             if not color_frame or not depth_frame:
#                 continue
            
#             # 將幀轉換為numpy數組
#             frame = np.asanyarray(color_frame.get_data())
            
#             # 偵測綠色圓形
#             circles, green_mask = detect_green_circles(color_frame, depth_frame)
            
#             # 如果檢測到圓形
#             if circles is not None:
#                 # 將圓形轉換為整數座標
#                 circles = np.uint16(np.around(circles))
                
#                 # 繪製圓形
#                 for i in circles[0, :]:
#                     # 取得深度值（單位：毫米）
#                     depth = depth_frame.get_distance(i[0], i[1])
                    
#                     # 繪製圓周
#                     cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    
#                     # 繪製圓心
#                     cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
                    
#                     # 標註座標（X, Y, Z）
#                     coord_text = f"({i[0]},{i[1]},{depth*1000:.0f}mm)"
#                     cv2.putText(frame, coord_text, 
#                                 (i[0]+10, i[1]+10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
#                                 (255, 255, 255), 1)
            
#             # 顯示影像
#             cv2.imshow('Green Circle Detection with Depth', frame)
            
#             # 顯示綠色遮罩
#             cv2.imshow('Green Mask', green_mask)
            
#             # 按 'q' 退出
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
    
#     finally:
#         # 關閉管線
#         pipeline.stop()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

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
                    
                    # 在綠色遮罩上繪製圓周
                    cv2.circle(green_mask_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    
                    # 繪製圓心
                    cv2.circle(green_mask_color, (i[0], i[1]), 2, (0, 0, 255), 3)
                    
                    # 標註座標（X, Y, Z）
                    coord_text = f"({i[0]},{i[1]},{depth*1000:.0f}mm)"
                    cv2.putText(green_mask_color, coord_text, 
                                (i[0]+10, i[1]+10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                (255, 255, 255), 1)
            
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