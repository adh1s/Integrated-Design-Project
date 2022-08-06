import numpy as np
import cv2 as cv
from os import listdir
from os.path import isfile, join
# Set path to dataset
path = 'distortion/imgs'

images = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

CHECKERBOARD = (6, 9)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(100)

#cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread('/Users/adhi/Desktop/IDP/distortion/imgs/600.jpg')
w,  h = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

mtx.tofile('distortion/cameramatrix.dat')
dist.tofile('distortion/distortionmatrix.dat')
newcameramtx.tofile('distortion/newcameramatrix.dat')