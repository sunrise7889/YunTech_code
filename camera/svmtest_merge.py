import cv2
import numpy as np
import pyrealsense2 as rs
import time
import joblib



# 初始化變數
points_2d = []
frame = None
depth_frame = None
prev_yellow_pos = None
prev_green_pos = None
prev_time = None
speed = 0

# 新增: 設定要顯示的前幀數量
prev_frame_index = 5  # 這個變數可以讓您設定想要顯示的是前幾幀的座標
# 新增: 儲存握把的歷史位置
handle_positions_history = []  # 儲存握把的前N幀位置

#SVMmodel = joblib.load("svclassifier.pkl") #讀取SVM模型

# 追蹤軌跡
trajectory_yellow = []
trajectory_green = []

# 設定 HSV 範圍
lower_yellow = np.array([20, 104, 100], dtype=np.uint8)
upper_yellow = np.array([28, 255, 183], dtype=np.uint8)

lower_green = np.array([40, 50, 50], dtype=np.uint8)
upper_green = np.array([90, 255, 255], dtype=np.uint8)

# 初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# 每 0.5 秒輸出一次
print_interval = 0.5
last_print_time = time.time()

print("等待 3 秒，穩定相機畫面...")
time.sleep(3)

# 滑鼠點擊回調函數
def select_points(event, x, y, flags, param):
    global points_2d, frame, depth_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        points_2d.append((x, y))  
        print(f"選取點: (x={x}, y={y})")

        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", frame)

def track_object(hsv_image, lower_HSV, upper_HSV, trajectory):
    """追蹤物件並回傳當前 (X, Y) 座標"""
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

# **開始點擊四個角來選取桌面**
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

# 設定俯視圖的尺寸
target_width, target_height = 600, 300  
target_points = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)

# 計算透視變換矩陣
image_points = np.array(points_2d, dtype=np.float32)
H = cv2.getPerspectiveTransform(image_points, target_points)
inv_H = np.linalg.inv(H)

# 定義桌面邊界
edges = [
    (target_points[0], target_points[1]),  # 上邊
    (target_points[1], target_points[2]),  # 右邊
    (target_points[2], target_points[3]),  # 下邊
    (target_points[3], target_points[0])   # 左邊
]

# 設定軌跡最大點數
MAX_TRAJECTORY = 50

# 確保歷史列表足夠長以存儲所需前幀
MAX_HISTORY = max(prev_frame_index + 1, 20)  # 至少保存 20 幀或更多幀

# **開始冰球與握把追蹤與碰撞檢測**
try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        warped_image = cv2.warpPerspective(color_image, H, (target_width, target_height), flags=cv2.INTER_LINEAR)

        # 將原始影像轉換為HSV
        hsv_original = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        # 將變換後的影像轉換為HSV
        hsv_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)

        # 在原始影像中追蹤黃色握把
        yellow_pos_original, _ = track_object(hsv_original, lower_yellow, upper_yellow, [])
        
        # 在變換後的影像中追蹤黃色握把
        current_handle_pos = None
        if yellow_pos_original:
            # 將原始座標轉換為俯視圖座標
            yellow_pos_homogeneous = np.array([yellow_pos_original[0], yellow_pos_original[1], 1])
            yellow_warped_pos = np.dot(H, yellow_pos_homogeneous)
            yellow_warped_pos = (int(yellow_warped_pos[0]/yellow_warped_pos[2]), 
                                int(yellow_warped_pos[1]/yellow_warped_pos[2]))
            
            # 儲存當前握把位置
            current_handle_pos = yellow_warped_pos
            
            # 添加當前位置到歷史記錄
            handle_positions_history.append(current_handle_pos)
            
            # 保持歷史記錄的長度不超過最大值
            if len(handle_positions_history) > MAX_HISTORY:
                handle_positions_history.pop(0)
            
            trajectory_yellow.append(yellow_warped_pos)
            if len(trajectory_yellow) > MAX_TRAJECTORY:
                trajectory_yellow.pop(0)
                
            # 在俯視圖中繪製黃色握把和軌跡
            cv2.circle(warped_image, yellow_warped_pos, 5, (0, 255, 255), -1)
            if len(trajectory_yellow) > 1:
                for i in range(1, len(trajectory_yellow)):
                    cv2.line(warped_image, trajectory_yellow[i-1], trajectory_yellow[i], (0, 255, 255), 2)
                    
            # 如果有足夠的歷史記錄，顯示指定前幀的位置
            if len(handle_positions_history) > prev_frame_index:
                prev_handle_pos = handle_positions_history[-prev_frame_index-1]
                # 在畫面上標註前幀位置
                cv2.circle(warped_image, prev_handle_pos, 5, (255, 0, 255), -1)  # 紫色圓點標記前幀位置
                cv2.putText(warped_image, f"前{prev_frame_index}幀", 
                           (prev_handle_pos[0] + 10, prev_handle_pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # 在變換後的影像中追蹤綠色冰球
        green_pos, _ = track_object(hsv_warped, lower_green, upper_green, [])
        
        # 綠色冰球處理
        if green_pos:
            current_position = green_pos
            current_time = time.time()
            
            # 追蹤綠色冰球軌跡
            trajectory_green.append(current_position)
            if len(trajectory_green) > MAX_TRAJECTORY:
                trajectory_green.pop(0)
                
            # 在俯視圖中繪製綠色冰球和軌跡
            cv2.circle(warped_image, current_position, 5, (0, 255, 0), -1)
            if len(trajectory_green) > 1:
                for i in range(1, len(trajectory_green)):
                    cv2.line(warped_image, trajectory_green[i-1], trajectory_green[i], (0, 255, 0), 2)

            # **轉換回原始影像座標**
            orig_pos = np.dot(inv_H, np.array([current_position[0], current_position[1], 1]))
            orig_x, orig_y = orig_pos[:2] / orig_pos[2]
            orig_x, orig_y = int(orig_x), int(orig_y)

            # **獲取深度值**
            z = int(depth_frame.get_distance(orig_x, orig_y) / depth_scale)

            # **速度計算**
            if prev_green_pos and prev_time:
                delta_time = current_time - prev_time
                delta_x = current_position[0] - prev_green_pos[0]
                delta_y = current_position[1] - prev_green_pos[1]
                distance = np.sqrt(delta_x**2 + delta_y**2)

                if distance < 2.3:
                    speed = 0
                else:
                    raw_speed = distance / delta_time
                    speed = 0.5 * raw_speed + (1 - 0.5) * speed  # 平滑處理

            prev_green_pos, prev_time = current_position, current_time

            # **檢測碰撞**
            collision = False
            for (p1, p2) in edges:
                # 計算點到線的距離
                x0, y0 = current_position
                x1, y1 = p1
                x2, y2 = p2

                # 計算線段長度
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if line_length == 0:
                    continue  # 避免除以 0

                # 計算垂直距離
                distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / line_length

                # 當冰球中心與邊界距離小於 5，視為碰撞
                if distance < 5:  
                    collision = True
                    print(f"碰撞偵測: (x={x0}, y={y0})")
                    cv2.circle(warped_image, (x0, y0), 10, (0, 0, 255), 3)  # 標記紅色碰撞區域

            # **畫出冰球位置與速度**
            cv2.putText(warped_image, f"Speed: {speed:.2f} ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # **畫出桌面邊界**
        for (p1, p2) in edges:
            cv2.line(warped_image, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 2)

        # 定期輸出座標信息
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            prev_handle_pos = None
            if len(handle_positions_history) > prev_frame_index:
                prev_handle_pos = handle_positions_history[-prev_frame_index-1]
                
            print(f"握把當前座標: {current_handle_pos}")
            print(f"握把前{prev_frame_index}幀座標: {prev_handle_pos}")
            print(f"冰球當前座標: {green_pos}, 速度: {speed:.2f}")
            print("-" * 50)
            last_print_time = current_time

        cv2.imshow("俯視圖", warped_image)
        #cv2.imshow("原始影像", color_image)
        
        # 按下 'q' 退出，按下 '[' 減少前幀數量，按下 ']' 增加前幀數量
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    pipeline.stop()