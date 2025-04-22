import cv2
import numpy as np
import pyrealsense2 as rs
import time
import joblib
# === 變數定義 ====
# left_bound    = 左邊界
# right_bound   = 右邊界
# top_bound     = 上邊界
# bottom_bound  = 下邊界
# center_line_x = 中線
# speed_cmps    = 球速
# cx_h , cy_h   = 握把當下XY軸
# prev_x , prev_y  = 握把上一幀XY軸
# cx_g , cy_g   = 冰球當下位置
# prev_handle = 定義要拿前多少幀座標

# === Read Model ==
pred = joblib.load("svm_model_no_scaler.pkl")

# === 參數設定 ===
SVMlock = 0
PUCK_RADIUS = 17  # 凍球半徑
target_width, target_height = 600, 300
px_to_cm = 120 / target_width  # 桌面寬度 120cm 對應 600px
PRINT_INTERVAL = 0.3
alpha = 0.5
PREV_HANDLE_INDEX = 5  # 要組回前第几幾年握把座標

lower_green = np.array([40, 50, 50])
upper_green = np.array([90, 255, 255])

points_2d = []
H = None
template = None
selected_roi = None
drawing = False
cx_g = cy_g = cx_h = cy_h = None
prev_g_pos = None
prev_time = None
speed_cmps = 0.0
last_print_time = time.time()
handle_buffer = []  # 儲存握把座標 buffer

# 設定邊界
margin = 5  # px 距離
left_bound = margin
right_bound = target_width - margin
top_bound = margin
bottom_bound = target_height - margin

# 分割右邊邊界
right_top_half = top_bound
right_bottom_half = (top_bound + bottom_bound) // 2

# 中線 x 位置
center_line_x = target_width // 2

# === 滑鼠事件：選角點 ===
def select_corners(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_2d) < 4:
        points_2d.append((x, y))
        print(f"點選角點: ({x}, {y})")

# === 滑鼠事件：選握把 ===
def roi_callback(event, x, y, flags, param):
    global x0, y0, drawing, selected_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x0, y0 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = x, y
        selected_roi = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

# === 初始化 RealSense ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
pipeline.start(config)

print("等待 3 秒穩定畫面...")
time.sleep(3)

# === 第一階段: 點選四角 ===
cv2.namedWindow("Select Corners")
cv2.setMouseCallback("Select Corners", select_corners)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    frame = np.asanyarray(color_frame.get_data()).copy()
    show = frame.copy()
    for pt in points_2d:
        cv2.circle(show, pt, 5, (0, 255, 0), -1)
    cv2.imshow("Select Corners", show)
    if len(points_2d) == 4 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("Select Corners")

# === Homography ===
target_pts = np.array([[0, 0], [target_width - 1, 0],
                       [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)
H = cv2.getPerspectiveTransform(np.array(points_2d, dtype=np.float32), target_pts)

# === 第二階段: 選擇握把 ===
cv2.namedWindow("Select Handle")
cv2.setMouseCallback("Select Handle", roi_callback)
print("請在促視圖上框選握把...")



while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    frame = np.asanyarray(color_frame.get_data())
    warped = cv2.warpPerspective(frame, H, (target_width, target_height))
    show = warped.copy()

    if selected_roi:
        x, y, w, h = selected_roi
        cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 255), 2)
        template = warped[y:y+h, x:x+w].copy()
        print("握把選取完成，開始追蹤...")
        break
    cv2.imshow("Select Handle", show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

cv2.destroyWindow("Select Handle")

# === 第三階段: 開始追蹤 ===
cv2.namedWindow("Tracking")

try:
    while True:
        frames = pipeline.wait_for_frames()
        frame = np.asanyarray(frames.get_color_frame().get_data())
        warped = cv2.warpPerspective(frame, H, (target_width, target_height))
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

        # === 顯示邊界 ===
        cv2.rectangle(warped, (left_bound, top_bound), (right_bound, bottom_bound), (0, 255, 255), 2)
        cv2.line(warped, (right_bound, right_bottom_half), (right_bound, bottom_bound), (0, 0, 255), 2)
        cv2.line(warped, (center_line_x, 0), (center_line_x, target_height), (0, 0, 255), 2)  # 中線

        # === 冰球 HSV 追蹤 ===
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx_g = M["m10"] / M["m00"]
                cy_g = M["m01"] / M["m00"]
                cv2.circle(warped, (int(cx_g), int(cy_g)), 5, (255, 0, 0), -1)

                now = time.time()
                if prev_g_pos is not None and prev_time is not None:
                    dx = cx_g - prev_g_pos[0]
                    dy = cy_g - prev_g_pos[1]
                    dt = now - prev_time
                    if dt > 0:
                        dist_px = np.sqrt(dx**2 + dy**2)
                        if dist_px < 0.5:
                            new_speed = 0
                        else:
                            dist_cm = dist_px * px_to_cm
                            new_speed = dist_cm / dt
                        speed_cmps = alpha * speed_cmps + (1 - alpha) * new_speed
                        if np.isnan(speed_cmps) or speed_cmps > 1000:
                            speed_cmps = 0
                        elif speed_cmps < 0.3:
                            speed_cmps = 0
                prev_g_pos = (cx_g, cy_g)
                prev_time = now

                

        # === 握把追蹤 ===
        if template is not None:
            res = cv2.matchTemplate(warped, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.5:
                top_left = max_loc
                bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
                cv2.rectangle(warped, top_left, bottom_right, (0, 0, 255), 2)
                cx_h = top_left[0] + template.shape[1] // 2
                cy_h = top_left[1] + template.shape[0] // 2
                cv2.circle(warped, (cx_h, cy_h), 5, (255, 0, 0), -1)

                handle_buffer.append((cx_h, cy_h))
                if len(handle_buffer) > PREV_HANDLE_INDEX:
                    prev_handle = handle_buffer[-PREV_HANDLE_INDEX - 1]
                else:
                    prev_handle = None           
        
        if prev_handle is not None:
            prev_x ,prev_y = prev_handle
            if (cx_h > prev_x) and (abs(cx_h - prev_x) > 10) and (SVMlock == 0):
                input_data = [[cx_h, cy_h, prev_x, prev_y, cx_g, cy_g]]
                prediction = pred.predict(input_data)
                print(f"Result: {prediction[0]}")
                SVMlock = 1  # 避免重複預測
            if cx_h<100:
                SVMlock = 0
                # =======策略選擇========
                if speed_cmps > 50:
                # #防守
                    print("Defence")
                # if speed_cmps < 50:
                # #進攻
                #     print("Attack")


        cv2.imshow("Tracking", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
