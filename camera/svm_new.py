import cv2
import numpy as np
import pyrealsense2 as rs

# 初始化變數
template = None
selected_roi = None
drawing = False
x0, y0 = -1, -1

def mouse_callback(event, x, y, flags, param):
    global x0, y0, drawing, selected_roi, template

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x0, y0 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = x, y
        selected_roi = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

# 初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
pipeline.start(config)

cv2.namedWindow("Tracking")
cv2.setMouseCallback("Tracking", mouse_callback)

print("請用滑鼠框選握把，選完後開始自動追蹤...")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # 抓取 template
        if selected_roi and template is None:
            x, y, w, h = selected_roi
            template = frame[y:y+h, x:x+w].copy()
            print("模板選取完成，開始追蹤...")

        if template is not None:
            res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > 0.5:
                top_left = max_loc
                bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

                # 中心點
                cx = top_left[0] + template.shape[1] // 2
                cy = top_left[1] + template.shape[0] // 2
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                # Print 中心點
                print(f"握把中心座標: ({cx}, {cy})")

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
