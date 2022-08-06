import cv2 as cv
import numpy as np
from main.util import detect_dropoff
# DROP OFF DETECTION FUNCTION TESTING+ITERATION SCRIPT

stream = cv.VideoCapture('http://localhost:8081/stream/video.mjpeg')

mtx = np.fromfile('distortion/cameramatrix.dat', dtype=float)
dist = np.fromfile('distortion/distortionmatrix.dat')
newmtx = np.fromfile('distortion/newcameramatrix.dat')
mtx = np.reshape(mtx, (3, 3))
dist = np.reshape(dist, (1,5))
newmtx = np.reshape(newmtx, (3,3))

r, f = stream.read()
f = cv.undistort(f, mtx, dist, None, newmtx)
f = f[:, 200:800]
drop_off = detect_dropoff(stream)

cv.circle(f, drop_off[0], 0,(255, 0,0),5)
cv.circle(f, drop_off[1], 0,(0, 255,0),5)
cv.circle(f, drop_off[2], 0,(0,0,255),5)
cv.circle(f, drop_off[3], 0,(255, 255, 0),5)
cv.imshow('Drop-off', f)
cv.waitKey()