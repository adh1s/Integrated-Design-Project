import cv2 as cv
import numpy as np
from main.util import detect_block
# BLOCK DETECTION FUNCTION TESTING+ITERATION SCRIPT

stream = cv.VideoCapture('http://localhost:8081/stream/video.mjpeg')

mtx = np.fromfile('distortion/cameramatrix.dat', dtype=float)
dist = np.fromfile('distortion/distortionmatrix.dat')
newmtx = np.fromfile('distortion/newcameramatrix.dat')
mtx = np.reshape(mtx, (3, 3))
dist = np.reshape(dist, (1,5))
newmtx = np.reshape(newmtx, (3,3))

while True:
    r, f = stream.read()
    f = cv.undistort(f, mtx, dist, None, newmtx)
    f = f[:, 200:800]
    block_coordinates = detect_block(f)
    if block_coordinates != None:
        cv.circle(f, block_coordinates, 0,(0, 0, 255),5)
        cv.imshow('Frame', f)
        cv.waitKey()