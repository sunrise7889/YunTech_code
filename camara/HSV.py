import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 建立 RealSense 管線
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 啟動相機
    pipeline.start(config)
    
    try:
        while True:
            # 等待下一個影格
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # 轉換影像為 NumPy 陣列
            color_image = np.asanyarray(color_frame.get_data())
            
            # 顯示影像
            cv2.imshow('RealSense RGB Stream', color_image)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # 停止相機
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
