'''
import cv2
import numpy as np
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Defining the world coordinates for 3D points
objp = np.zeros((8*5,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1, 2)
objp = objp*27
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 

# Extracting path of individual image stored in a given directory
images = glob.glob('C:/Users/nihao/Desktop/pptG/withoutNorData/2024_07_18/picture/*.png')
for fname in images:
    
    img = cv2.imread(fname)
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, (8,5), None)

    if ret == True:
        
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8,5), corners2,ret)
        cv2.imshow(fname,img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
# cameraMatrix = np.array([[380.234, 0, 322.568], [0, 380.234, 241.794], [0, 0, 1]], dtype=np.float32)  
# distCoeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#ret, rvecs, tvecs = cv2.solvePnP(objpoints, imgpoints, gray.shape[::-1],cameraMatrix,distCoeffs)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)
'''
#https://blog.csdn.net/qq_51116518/article/details/126065796
#計算咖啡色矩陣
#========================================================================================================
import cv2
import numpy as np
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Defining the world coordinates for 3D points
#校正版內角點數量(長*寬)
objp = np.zeros((8*5,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1, 2)
#校正版小格子邊長(mm)
objp = objp*27

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 

# Extracting path of individual image stored in a given directory
images = glob.glob('D:/2024_07_20/*.png')
for fname in images:
    
    img = cv2.imread(fname)
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,5), None)

    if ret == True:
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8,5), corners2, ret)
        cv2.imshow(fname, img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# Define your camera matrix and distortion coefficients here
#內部參數
cameraMatrix = np.array([[604.85, 0, 319.963], [0, 604.728, 237.925], [0, 0, 1]], dtype=np.float32)
distCoeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

# Instead of calibrateCamera, use solvePnP if you have the camera matrix and distortion coefficients
rvecs, tvecs = [], []

for objp, imgp in zip(objpoints, imgpoints):
    ret, rvec, tvec = cv2.solvePnP(objp, imgp, cameraMatrix, distCoeffs)
    if ret:
        rvecs.append(rvec)
        tvecs.append(tvec)

print("Camera matrix : \n")
print(cameraMatrix)
print("Distortion coefficients: \n")
print(distCoeffs)
print("Rotation vectors: \n")
print(rvecs)
print("Translation vectors: \n")
print(tvecs)
