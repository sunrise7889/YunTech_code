import cv2
import numpy as np
import pyrealsense2 as rs
import time

# 初始化變數
points_2d = []
frame = None
depth_frame = None
prev_green_pos = None
prev_time = None
speed = 0

prev_frame_index = 5
handle_positions_history = []
trajectory_red = []
trajectory_green = []

# HSV 範圍設定
lower_red1 = np.array([0, 120, 70], dtype=np.uint8)
upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
lower_red2 = np.array([170, 120, 70], dtype=np.uint8)
upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

lower_green = np.array([40, 50, 50], dtype=np.uint8)
upper_green = np.array([90, 255, 255], dtype=np.uint8)

# 初始化 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
video_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()

print("等待 3 秒，穩定相機畫面...")
time.sleep(3)

# 滑鼠選取四個點進行俯視校正
def select_points(event, x, y, flags, param):
    global points_2d, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points_2d.append((x, y))
        print(f"選取點: (x={x}, y={y})")
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", frame)

# 偵測紅色圓形握把
def detect_red_circle(hsv_image, max_radius_px=35, draw_on_img=None):
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.GaussianBlur(mask, (9, 9), 2)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=100, param2=15, minRadius=5, maxRadius=max_radius_px)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        if draw_on_img is not None:
            cv2.circle(draw_on_img, (x, y), r, (0, 0, 255), 2)
            cv2.circle(draw_on_img, (x, y), 2, (255, 255, 255), 2)
        return (x, y), r, mask
    return None, None, mask

# HSV 綠色冰球追蹤
def track_object(hsv_image, lower_HSV, upper_HSV, trajectory):
    mask = cv2.inRange(hsv_image, lower_HSV, upper_HSV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            trajectory.append((cX, cY))
            return (cX, cY), mask
    return None, mask

# 點選四點做俯視校正
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        continue
    frame = np.asanyarray(color_frame.get_data()).copy()
    cv2.imshow("Select Points", frame)
    cv2.setMouseCallback("Select Points", select_points)

    if len(points_2d) == 4:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

target_width, target_height = 600, 300
target_points = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)
image_points = np.array(points_2d, dtype=np.float32)
H = cv2.getPerspectiveTransform(image_points, target_points)
inv_H = np.linalg.inv(H)

edges = [
    (target_points[0], target_points[1]),
    (target_points[1], target_points[2]),
    (target_points[2], target_points[3]),
    (target_points[3], target_points[0])
]

MAX_TRAJECTORY = 50
MAX_HISTORY = max(prev_frame_index + 1, 20)

# 主迴圈
try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        warped_image = cv2.warpPerspective(color_image, H, (target_width, target_height), flags=cv2.INTER_LINEAR)
        hsv_original = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        hsv_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)

        # 紅色握把偵測
        red_center, radius, _ = detect_red_circle(hsv_original, draw_on_img=color_image)
        current_handle_pos = None
        if red_center:
            red_homo = np.array([red_center[0], red_center[1], 1])
            red_warped_pos = np.dot(H, red_homo)
            red_warped_pos = (int(red_warped_pos[0]/red_warped_pos[2]), int(red_warped_pos[1]/red_warped_pos[2]))
            current_handle_pos = red_warped_pos
            handle_positions_history.append(current_handle_pos)
            if len(handle_positions_history) > MAX_HISTORY:
                handle_positions_history.pop(0)

            trajectory_red.append(current_handle_pos)
            if len(trajectory_red) > MAX_TRAJECTORY:
                trajectory_red.pop(0)

            cv2.circle(warped_image, current_handle_pos, 5, (0, 0, 255), -1)
            for i in range(1, len(trajectory_red)):
                cv2.line(warped_image, trajectory_red[i-1], trajectory_red[i], (0, 0, 255), 2)

            if len(handle_positions_history) > prev_frame_index:
                prev_handle_pos = handle_positions_history[-prev_frame_index-1]
                cv2.circle(warped_image, prev_handle_pos, 5, (100, 100, 255), -1)

        # 綠色冰球追蹤
        green_pos, _ = track_object(hsv_warped, lower_green, upper_green, [])
        if green_pos:
            current_position = green_pos
            current_time = time.time()

            trajectory_green.append(current_position)
            if len(trajectory_green) > MAX_TRAJECTORY:
                trajectory_green.pop(0)

            cv2.circle(warped_image, current_position, 5, (0, 255, 0), -1)
            for i in range(1, len(trajectory_green)):
                cv2.line(warped_image, trajectory_green[i-1], trajectory_green[i], (0, 255, 0), 2)

            orig_pos = np.dot(inv_H, np.array([current_position[0], current_position[1], 1]))
            orig_x, orig_y = orig_pos[:2] / orig_pos[2]
            orig_x, orig_y = int(orig_x), int(orig_y)
            z = int(depth_frame.get_distance(orig_x, orig_y) / depth_scale)

            if prev_green_pos and prev_time:
                delta_time = current_time - prev_time
                delta_x = current_position[0] - prev_green_pos[0]
                delta_y = current_position[1] - prev_green_pos[1]
                distance = np.sqrt(delta_x**2 + delta_y**2)
                if distance < 2.3:
                    speed = 0
                else:
                    raw_speed = distance / delta_time
                    speed = 0.5 * raw_speed + 0.5 * speed

            prev_green_pos, prev_time = current_position, current_time
            cv2.putText(warped_image, f"Speed: {speed:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        for (p1, p2) in edges:
            cv2.line(warped_image, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 2)

        cv2.imshow("俯視圖", warped_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    pipeline.stop()
