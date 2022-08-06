from copy import deepcopy
import numpy as np
import math
import cv2 as cv

mtx = np.fromfile('distortion/cameramatrix.dat', dtype=float)
dist = np.fromfile('distortion/distortionmatrix.dat')
newmtx = np.fromfile('distortion/newcameramatrix.dat')
mtx = np.reshape(mtx, (3, 3))
dist = np.reshape(dist, (1,5))
newmtx = np.reshape(newmtx, (3,3))

def intermediates(p1: tuple, p2: tuple, nb_points: int = 1):
    """p1: first coordinate
       p2: second coordinate
       returns nb_points equally spaced points interpolated with p1 and p2"""
    # E.g 8 intermediate points: 8+1=9 spaces between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [(int(p1[0] + i * x_spacing), int(p1[1] +  i * y_spacing))
            for i in range(1, nb_points+1)]

def calculate_required_translation(aruco_tag_detection, destination: tuple):
    """apriltag_tag_detection:  each result from the detector in a frame - data struct in https://pypi.org/project/apriltag/
       destination:   coordinates of the next destination in the trip
       returns translation distance (in m), robot vector (direction robot is facing), 
       translation vector (direction of the required translation) -- (unit vectors) """
    (ptA, ptB, _, _) = aruco_tag_detection.corners
    ptB = (int(ptB[0]), int(ptB[1]))
    ptA = (int(ptA[0]), int(ptA[1]))

    # Draw the center (x, y)-coordinates of the AprilTag
    (cX, cY) = (int(aruco_tag_detection.center[0]), 
                int(aruco_tag_detection.center[1]))
    
    x_comp = ((ptB[0] + ptA[0])/ 2) - cX
    y_comp = cY - ((ptB[1] + ptA[1])/2)
    robot_vector = np.array([x_comp, y_comp, 0])
    robot_vector = robot_vector/np.linalg.norm(robot_vector)

    translation_x = destination[0] - cX
    translation_y = cY - destination[1]
    translation_vector = np.array([translation_x, translation_y, 0])
    translation_pixel_distance = np.linalg.norm(translation_vector)
    translation_vector = translation_vector/np.linalg.norm(translation_vector) 

    # Basic pixel to distance calibration
    front_edge = np.array([(ptB[0] - ptA[0]), (ptB[1] - ptA[1])])
    front_edge_pixel_distance = np.linalg.norm(front_edge) 
    front_edge_length = 0.093 
    translation_distance = ((translation_pixel_distance / front_edge_pixel_distance) * front_edge_length) 
    
    return translation_distance, translation_vector, robot_vector

def calculate_angle (x: np.array, y: np.array) -> float:
    """x: robot vector
       y: path vector 
       returns angle (negative for turn right, positive for turn left"""
    cross_product = np.cross(x, y) 
    dot_product = np.dot(x, y)
    
    # Fix math domain error from bit precision errors
    if cross_product[2] > 1: 
        cross_product[2] = 1
    if cross_product[2] < -1:
        cross_product[2] = -1
    
    radians = math.asin(cross_product[2])
    angle = radians * 180/math.pi

    # Fixing some quadrant funkiness
    if dot_product < 0 and angle < 0:
        angle = -angle
        angle = -180 + angle
    elif dot_product < 0 and angle > 0:
        angle = -angle
        angle = 180 + angle
    elif dot_product < 0 and angle == 0:
        angle += 180
    return angle 

def detect_block(f):
    """f: OpenCV undisorted+cropped frame --- Shape is (760, 1016, 3)
       returns (x, y) or None of the block"""
    f = f[535:675, 200:335] # Tight region of interest
    f_hsv=cv.cvtColor(f, cv.COLOR_BGR2HSV)
    lower_red = np.array([0,50,50]) # Lower scale mask (0-10)
    upper_red = np.array([20,255,255]) 
    mask0 = cv.inRange(f_hsv, lower_red, upper_red)
    lower_red = np.array([170,50,50]) # Upper scale mask (170-180)
    upper_red = np.array([180,255,255])
    mask1 = cv.inRange(f_hsv, lower_red, upper_red)
    mask = mask0+mask1 # Join masks
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
            return center
    return None

def detect_intersections(f):
    """f: OpenCV undistorted+cropped frame -
       returns [(x1, y1), (x2, y2)] - the coordinates of first and second 
       intersection - returns [None, None] if not found"""
    check1 = cv.imread('/Users/adhi/Desktop/IDP/main/features/checkpoint1.png',0)
    check2 = cv.imread('/Users/adhi/Desktop/IDP/main/features/checkpoint2.png',0) 
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
    
    if (dst_pt1 and dst_pt2) and (int(dst_pt2[0][0]) < 380 and int(dst_pt2[0][1]) > 300):
        return [(int(dst_pt1[0][0]), int(dst_pt1[0][1])), (int(dst_pt2[0][0]), int(dst_pt2[0][1]))]
    else:
        return [None, None]

def detect_dropoff(s) -> list:
    """s:stream
       returns [(x, y), (x,y) ...] of the drop_off_locations - [Blue (top), Blue (bottom), Red (bottom), Red (top)] """
    frame_counter = 0
    threshold = 230 # Threshold that works well 
    while frame_counter <= 100:
        r, f = s.read()
        frame_counter += 1
        f = cv.undistort(f, mtx, dist, None, newmtx)
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
        #blue is between 115 and 140, 0.5 is for safety (up to half the pixels)
        #red between 130 and 180, 2 is for up to double the pixels
        min_area = 50 
        max_area = 220 

        centers = []
        for c in cnts_b:
            area = cv.contourArea(c)
            blue_centers = []
            if area > min_area and area < max_area:
                (x,y),_ = cv.minEnclosingCircle(c)
                center = (int(x) + 280 ,int(y) + 140)
                blue_centers.append(center)
            if len(blue_centers) == 2:
                centers += blue_centers

        for c in cnts_r:
            area = cv.contourArea(c)
            red_centers = []
            if area > min_area and area < max_area:
                (x,y),_ = cv.minEnclosingCircle(c)
                center = (int(x) + 400 ,int(y) + 300)
                red_centers.append(center)
            if len(red_centers) == 2:
                centers += red_centers
        
        if len(centers) == 4:
            centers.sort(key=lambda y: y[0])
            # Sort to return coordinates in the required order
            return [centers[0], centers[1], centers[3], centers[2]] 
        
    # Failsafe for if the drop off points are not found in the first 100 frames
    return [(313, 222), (375, 166), (525, 331), (467, 389)] 

def forward_path(s) -> list:
    """s:stream
       returns [(x, y), (x,y) ...] of the forward_path"""
    # Failsafe for if the block or intersection route is not found in the first 100 frames
    block = (83, 590)
    path = [(421, 275), (288, 396), (156, 522)]
    frame_counter = 0
    
    while frame_counter <= 100:
        r, f = s.read() 
        frame_counter += 1
        f = cv.undistort(f, mtx, dist, None, newmtx)
        f = f[:, 200:800]
        block_detection = detect_block(f)
        intersections = detect_intersections(f)
        if block_detection != None:
            block = deepcopy(block_detection)
        if None not in intersections:
            path = [intersections[0], intermediates(intersections[0], intersections[1]), intersections[1]]
    
    path += block 
    return path
