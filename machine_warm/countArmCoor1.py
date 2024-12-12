import numpy as np
import cv2
from math import *
import math

# 用于根据欧拉角计算旋转矩阵
def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = np.dot(np.dot(Rz, Ry), Rx)
    return R
 
 
# 用于根据位姿计算变换矩阵
def pose_robot(z, y, x, Tx, Ty, Tz):  # 注意输入顺序！！！！！！
    """
    Args:
        x: x轴旋转角度
        y: y轴旋转角度
        z: z轴旋转角度
        Tx: Tx为平移横坐标
        Ty: Ty为平移纵坐标
        Tz: Tz为平移高坐标
    Returns: 旋转矩阵与平移矩阵的叠加，即两坐标系之间的转换矩阵
    """
    thetaX = x / 180 * pi
    thetaY = y / 180 * pi
    thetaZ = z / 180 * pi
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)
    t = np.array([[Tx], [Ty], [Tz]])
    RT1 = np.column_stack([R, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
    return RT1
 
 
# 用来从棋盘格图片得到相机外参
def get_RT_from_chessboard(rot_vector, trans):
    rotMat = cv2.Rodrigues(rot_vector)[0]
    t = np.array([[trans[0], trans[1], trans[2]]]).T
    RT = np.column_stack((rotMat, t))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    return RT
 
 
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
 
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
 
    singular = sy < 1e-6
 
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
 
    return np.array([x, y, z])

# 从相机坐标系到末端执行器坐标系的变换矩阵
RT_cam_to_end = np.array([[-7.20186026e-03, -9.99968481e-01,  3.34211490e-03,  3.38646268e+01],
 [ 9.99973141e-01, -7.19726613e-03,  1.38461270e-03, -4.35234105e+01],
 [-1.36051497e-03,  3.35199692e-03,  9.99993457e-01, -3.49352404e+01],
 [  0.0,           0.0,          0.0,           1.0        ]])
#===========================================================================================================================
# 工件圆心的相机坐标
#原點教正要補償(+13,-19)
#相機原點workpiece_center_cam = np.array([0, 0, 406, 1]).T 計算出xyzabc，然後輸入到示教器，看手臂末端移到哪裡，再用相機軟體看兩個座標的相差值
workpiece_center_cam = np.array([8+13, -17-19, 385-50, 1]).T
#===========================================================================================================================


# 将工件圆心坐标转换到末端执行器坐标系中
workpiece_center_end = np.dot(RT_cam_to_end, workpiece_center_cam)
print("相機座標*RT")
print(workpiece_center_end)


#===========================================================================================================================
#綠色矩陣(設定原點)
end2robot = [239.372, 224.514, 584.375, 42.291, -10.139, 176.571]
#==========================================================================================================================
end2robot_RT = pose_robot(end2robot[3],end2robot[4],end2robot[5],end2robot[0],end2robot[1],end2robot[2])
print("變換矩陣 end2robot_RT")
print(end2robot_RT)

finalCoor = np.dot(end2robot_RT, workpiece_center_end)
print(finalCoor)
x_base, y_base, z_base = finalCoor[:3]
print("物體在機器人基座座標系中的座標:")
print(x_base, y_base, z_base)

# 提取旋转矩阵
R_base = end2robot_RT[:3, :3]
angles_base = rotationMatrixToEulerAngles(R_base)
#print("物體在基座座標系中的姿態 (RX, RY, RZ):")
#print(angles_base)

# 输出姿态角度，单位为度
angles_base_degrees = np.degrees(angles_base)
print("物體在基座座標系中的姿態 (RX, RY, RZ) in degrees:")
print(angles_base_degrees)