from xarm.wrapper import XArmAPI
import pandas as pd
import numpy as np
import cv2

T = np.load("arm\cam2arm_matrix.npy")
if T is not None:
    print("以載入轉換矩陣")
# === 初始化手臂 ===
arm = XArmAPI('192.168.1.160')
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)

camera = [882, 324, 899]
point = np.array(camera+[1]).reshape(4, 1)
point_arm = T @ point

arm_xyz = point_arm[:3].flatten()
print(arm_xyz)
arm.set_position(*arm_xyz, speed=10,wait=True)
