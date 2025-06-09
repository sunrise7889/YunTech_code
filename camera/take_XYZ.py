# === Eye-to-Hand 手眼校正完整模組 ===
# 功能：滑鼠選四點做 Homography → 自動抓棋盤角點 → 儲存相機角點 + 機械手臂 XYZABC（兩段式） → 計算轉換矩陣

import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R

# === 初始化手臂 ===
arm = XArmAPI('192.168.1.160')
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)
target_width, target_height = 600, 300

# === 初始化 RealSense ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
profile = pipeline.start(config)

# === 滑鼠點四點建立 Homography ===
clicked_points = []
H = None

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])

print("[請點擊畫面四個角點做俯視轉換]")
while True:
    frames = pipeline.wait_for_frames()
    img = np.asanyarray(frames.get_color_frame().get_data())
    temp_img = img.copy()
    for pt in clicked_points:
        cv2.circle(temp_img, tuple(pt), 5, (0, 255, 0), -1)
    cv2.imshow("Select Points", temp_img)
    cv2.setMouseCallback("Select Points", click_event)
    if cv2.waitKey(1) == ord('q') or len(clicked_points) == 4:
        break
cv2.destroyWindow("Select Points")

dst = np.array([[0, 0], [target_width - 1, 0],
                       [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)
H = cv2.getPerspectiveTransform(np.array(clicked_points, dtype=np.float32), dst)
print("✅ Homography 建立完成")

# === 棋盤格參數 ===
pattern_size = (6, 4)
square_size = 39
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= square_size

# === 儲存用 ===
Rs_cam2target, Ts_cam2target = [], []
Rs_base2gripper, Ts_base2gripper = [], []

# === 狀態：0 等待相機角點，1 等待手臂對準 ===
state = 0
print("[s] 儲存角點與手臂姿態配對（兩段式），[q] 離開")
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    img = np.asanyarray(color_frame.get_data())
    warped = cv2.warpPerspective(img, H, (600, 300))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        corners_sub = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.drawChessboardCorners(warped, pattern_size, corners_sub, ret)
        # 標記角點 0
        cv2.circle(warped, tuple(corners_sub[0][0].astype(int)), 8, (0, 0, 255), -1)
        cv2.putText(warped, "P0", tuple(corners_sub[0][0].astype(int) + np.array([5, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Chessboard", warped)

        key = cv2.waitKey(1)
        if key == ord('s'):
            if state == 0:
                retval, rvec, tvec = cv2.solvePnP(objp, corners_sub, np.eye(3), None)
                R_cam, _ = cv2.Rodrigues(rvec)
                Rs_cam2target.append(R_cam)
                Ts_cam2target.append(tvec)
                print("📌 已儲存相機角點資訊，請移動手臂末端對準紅點後再按 s")
                state = 1
            elif state == 1:
                pos = arm.get_position(is_radian=False)
                if not isinstance(pos, list) or len(pos) < 6:
                    print("❌ 無法取得有效手臂位置，pos =", pos)
                    continue
                xyz = np.array(pos[:3])
                rpy = R.from_euler('xyz', pos[3:6], degrees=True)
                Rs_base2gripper.append(rpy.as_matrix())
                Ts_base2gripper.append(xyz.reshape(3,1))
                print(f"✅ 已儲存第 {len(Rs_cam2target)} 筆對應點（相機 + 手臂）")
                state = 0
        elif key == ord('q'):
            break
    else:
        cv2.imshow("Chessboard", warped)
        if cv2.waitKey(1) == ord('q'):
            break

pipeline.stop()
cv2.destroyAllWindows()

# === 執行校正 ===
R_cam2arm, t_cam2arm = cv2.calibrateHandEye(
    Rs_base2gripper, Ts_base2gripper,
    Rs_cam2target, Ts_cam2target,
    method=cv2.CALIB_HAND_EYE_TSAI
)
T = np.eye(4)
T[:3, :3] = R_cam2arm
T[:3, 3] = t_cam2arm.flatten()
np.save("cam2arm_matrix.npy", T)
print("\n✅ 校正完成，轉換矩陣 cam2arm_matrix.npy 已儲存：")
print(T)
