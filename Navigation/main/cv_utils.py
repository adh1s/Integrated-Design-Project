import numpy as np
import cv2 as cv
from utils import coordinate

def detect_block(f) -> coordinate:
    """f: undisorted, cropped frame
    returns coordinate of the block or None"""
    f = f[535:675, 200:335] # Tight region of interest
    f_hsv=cv.cvtColor(f, cv.COLOR_BGR2HSV)
    lower_red = np.array([0,50,50]) # Lower scale mask (0-10)
    upper_red = np.array([20,255,255]) 
    mask0 = cv.inRange(f_hsv, lower_red, upper_red)
    lower_red = np.array([170,50,50]) # Upper scale mask (170-180)
    upper_red = np.array([180,255,255])
    mask1 = cv.inRange(f_hsv, lower_red, upper_red)
    mask = mask0 + mask1 # Join masks
    # Set img to zero everywhere except the mask
    f[np.where(mask==0)] = 0
    gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv.filter2D(blur, -1, sharpen_kernel)

    thresh = cv.threshold(sharpen,20,255, cv.THRESH_BINARY_INV)[1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
    close = cv.bitwise_not(close)

    cnts = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Expect the block to be within this pixel range
    min_area = 25
    max_area = 150 
    # Only consider frame if there is one contour
    if len(cnts) == 1: 
        cnt = cnts[0]
        area = cv.contourArea(cnt)
        if area > min_area and area < max_area: 
            (x,y), _ = cv.minEnclosingCircle(cnt)
            center = (int(x),int(y) + 535)
            return coordinate(center[0], center[1])
    return None

def detect_intersections(f) -> list:
    """f: undistorted, cropped frame
    returns coordinates of intersections or None"""
    check1 = cv.imread('/Users/adhi/Documents/Integrated_Design_Project/main/features/checkpoint1.png',0)
    check2 = cv.imread('/Users/adhi/Documents/Integrated_Design_Project/main/features/checkpoint2.png',0) 
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    _, des1 = sift.detectAndCompute(check1,None)
    _, des2 = sift.detectAndCompute(check2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    kp3, des3 = sift.detectAndCompute(f,None)
    matches_1 = flann.knnMatch(des1,des3,k=2)
    matches_2 = flann.knnMatch(des2,des3,k=2)
    # Store all the good matches as per Lowe's ratio test.
    good_1 = []
    good_2 = []
    
    for m,n in matches_1:
        if m.distance < 0.7*n.distance:
            good_1.append(m)
    
    for m,n in matches_2:
        if m.distance < 0.7*n.distance:
            good_2.append(m)

    dst_pt1 = [kp3[m.trainIdx].pt for m in good_1]
    dst_pt2 = [kp3[m.trainIdx].pt for m in good_2]
    
    # Sanity checks
    if (dst_pt1 and dst_pt2) and (int(dst_pt2[0][0]) < 380 and int(dst_pt2[0][1]) > 300):
        return [coordinate(int(dst_pt1[0][0]), int(dst_pt1[0][1])), coordinate(int(dst_pt2[0][0]), int(dst_pt2[0][1]))]
    return None

def detect_dropoff(f) -> list:
    """f: undisorted, cropped frame
    returns coordinates of drop-off locations - [Blue, Blue, Red, Red] - or None"""
    threshold = 230 

    f = f[:, 200:800]
    blues = f[140:250, 280:430]
    reds = f[300:420, 400:550]

    gray_b = cv.cvtColor(blues, cv.COLOR_BGR2GRAY)
    gray_r = cv.cvtColor(reds, cv.COLOR_BGR2GRAY)
    blur_b = cv.medianBlur(gray_b, 5)
    blur_r = cv.medianBlur(gray_r, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen_b = cv.filter2D(blur_b, -1, sharpen_kernel)
    sharpen_r = cv.filter2D(blur_r, -1, sharpen_kernel)
    
    thresh_b = cv.threshold(sharpen_b,threshold,255, cv.THRESH_BINARY_INV)[1]
    thresh_r = cv.threshold(sharpen_r,threshold,255, cv.THRESH_BINARY_INV)[1]
    thresh_b = cv.bitwise_not(thresh_b)
    thresh_r = cv.bitwise_not(thresh_r)

    cnts_r = cv.findContours(thresh_r, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts_r = cnts_r[0] if len(cnts_r) == 2 else cnts_r[1]

    cnts_b = cv.findContours(thresh_b, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts_b = cnts_b[0] if len(cnts_b) == 2 else cnts_b[1]

    min_area = 50 
    max_area = 220 

    centers = []
    for c in cnts_b:
        area = cv.contourArea(c)
        blue_centers = []
        if area > min_area and area < max_area:
            (x,y),_ = cv.minEnclosingCircle(c)
            center = (int(x) + 280, int(y) + 140)
            blue_centers.append(center)
        if len(blue_centers) == 2:
            centers += blue_centers

    for c in cnts_r:
        area = cv.contourArea(c)
        red_centers = []
        if area > min_area and area < max_area:
            (x,y),_ = cv.minEnclosingCircle(c)
            center = (int(x) + 400, int(y) + 300)
            red_centers.append(center)
        if len(red_centers) == 2:
            centers += red_centers
    
    if len(centers) == 4:
        centers.sort(key=lambda y: y[0])
        return [coordinate(centers[0][0], centers[0][1]),
                coordinate(centers[1][0], centers[1][1]),
                coordinate(centers[3][0], centers[3][1]),
                coordinate(centers[2][0], centers[2][1])]
    return None