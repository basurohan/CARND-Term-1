import cv2
import glob
import numpy as np
import pickle

# Arrays to store object points and image points
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points like (0,0,0), (1,0,0), (2,0,0) ....., (8,5,0)
# nx=9, ny=6 - as per calibration images provided
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2) # x, y cordinates

images = glob.glob("camera_cal/*.jpg")

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessbord corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If corners found, set object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw corners and save images
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        fileName = "output_images/corners_found/calibration_corners"+str(idx)+".jpg"
        cv2.imwrite(fileName, img)

img = cv2.imread("camera_cal/calibration1.jpg")
img_size = (img.shape[1], img.shape[0])

# Perform camera calibration to get - camera marix and distortion coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save the camera matrix and the distortion coefficients for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("calibration_pickle.p", "wb"))