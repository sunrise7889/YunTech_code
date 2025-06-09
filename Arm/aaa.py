# 匯入 CSV 中的 (相機座標 XYZ, 手臂座標 XYZ)，使用仿射轉換估計 cam2arm_matrix.npy
# 不使用姿態，僅使用 estimateAffine3D 快速取得 4x4 轉換矩陣（近似）

import pandas as pd
import numpy as np
import cv2

# === 讀取 CSV 資料 ===
csv_path = 'Arm\eye2hand_2.csv'  # 修改為你的檔案名稱
df = pd.read_csv(csv_path)

# 檢查必要欄位
required_cols = ["Cam_X", "Cam_Y", "Cam_Z", "Arm_X", "Arm_Y", "Arm_Z"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV 缺少欄位：{col}")

# === 準備點資料 ===
cam_pts = df[["Cam_X", "Cam_Y", "Cam_Z"]].to_numpy().astype(np.float32)
arm_pts = df[["Arm_X", "Arm_Y", "Arm_Z"]].to_numpy().astype(np.float32)

# Reshape 成 (N,1,3) 符合 estimateAffine3D 格式
cam_pts = cam_pts.reshape(-1, 1, 3)
arm_pts = arm_pts.reshape(-1, 1, 3)

# === 執行仿射校正 ===
retval, affine, inliers = cv2.estimateAffine3D(cam_pts, arm_pts)

if not retval:
    raise RuntimeError("❌ estimateAffine3D 失敗，請確認資料是否有空間變化")

# 補齊成 4x4 homogeneous matrix
T = np.eye(4)
T[:3, :] = affine
T[2, 3] -= 14

np.save("matrix_5.npy", T)
print("\n✅ 仿射校正完成（不含旋轉姿態），轉換矩陣 cam2arm_matrix.npy 已儲存：")
print(T)
