import cv2
import numpy as np
import pyrealsense2 as rs
import time

# 設定印出間隔時間（秒）
PRINT_INTERVAL = 0.3
last_print_time = time.time()

# HSV 綠色範圍
lower_green = np.array([40, 50, 50])
upper_green = np.array([90, 255, 255])

# 初始化變數
template = None
selected_roi = None
drawing = False
x0, y0 = -1, -1
cx_h, cy_h = None, None  # 握把中心點
cx_g, cy_g = None, None  # 綠球中心點

def mouse_callback(event, x, y, flags, param):
    global x0, y0, drawing, selected_roi, template
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x0, y0 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = x, y
        selected_roi = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

# 初始化 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
pipeline.start(config)

cv2.namedWindow("Tracking")
cv2.setMouseCallback("Tracking", mouse_callback)

# 啟動延遲
print("系統啟動中，請等待畫面穩定...")
time.sleep(3)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- 綠色冰球追蹤 ---
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx_g = int(M["m10"] / M["m00"])
                cy_g = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx_g, cy_g), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"({cx_g}, {cy_g})", (cx_g + 10, cy_g),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- 握把追蹤 ---
        if selected_roi and template is None:
            x, y, w, h = selected_roi
            template = frame[y:y+h, x:x+w].copy()
            print("握把模板選取完成，開始追蹤...")

        if template is not None:
            res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.5:
                top_left = max_loc
                bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

                cx_h = top_left[0] + template.shape[1] // 2
                cy_h = top_left[1] + template.shape[0] // 2
                cv2.circle(frame, (cx_h, cy_h), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"({cx_h}, {cy_h})", (cx_h + 10, cy_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- 控制 print 頻率 ---
        current_time = time.time()
        if current_time - last_print_time > PRINT_INTERVAL:
            if cx_h is not None and cy_h is not None:
                print(f"握把中心座標: ({cx_h}, {cy_h})", end='  ')
            if cx_g is not None and cy_g is not None:
                print(f"綠色冰球中心座標: ({cx_g}, {cy_g})")
            last_print_time = current_time

        # 顯示畫面
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
