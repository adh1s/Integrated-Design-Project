import apriltag
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from util import *

blocks_to_attempt = 4 # Number of blocks to return in the competition
comm_interval = 10 # Number of frames between sending information to the Arduino

mqttBroker = "broker.hivemq.com"
client = mqtt.Client("Python")
client.connect(mqttBroker) 
stream = cv2.VideoCapture('http://localhost:8081/stream/video.mjpeg')

# Load camera properties 
mtx = np.fromfile('distortion/cameramatrix.dat', dtype=float)
dist = np.fromfile('distortion/distortionmatrix.dat')
newmtx = np.fromfile('distortion/newcameramatrix.dat')
mtx = np.reshape(mtx, (3, 3))
dist = np.reshape(dist, (1,5))
newmtx = np.reshape(newmtx, (3,3))

# Try detect landmarks of the arena at run-time
original_forward_path = forward_path(stream) 
starting_location = (532, 175) # Can hard-code - only need a coarse starting location
intersection_1 = original_forward_path[0]
ramp = original_forward_path[1]
intersection_2 = original_forward_path[2]
# Interpolate between forward path - bottom ramp helps with turning close to ramp
bottom_ramp = intermediates(intersection_2, ramp, 4)[1] 

drop_off_locations = detect_dropoff(stream) # Order - [Blue (top), Blue (bottom), Red (bottom), Red (top)]
# Checkpoint right before dropping off in the 'top' and 'bottom' boxes - helps with turning 
bottom_drop_off = intermediates(original_forward_path[0], starting_location, 7)[0] 
top_drop_off = (bottom_drop_off[0] + 66, bottom_drop_off[1] - 60) 

# Blue and Red blocks picked up
blue = 0
red = 0

while (red + blue) < blocks_to_attempt: # Loop over this for each block to be retrieved
    frame_counter = 0 
    if (red + blue) == 0: # Initialise path for the first block
        destinations = [intersection_1, ramp, intersection_2, detect_block(stream)]
    else: # Initialise path for the further blocks
        destinations = [ramp, intersection_2, detect_block(stream)]
    while len(destinations) > 0:
        r, f = stream.read()
        frame_counter += 1
        if frame_counter % 2 == 0: # Calculations every 2 frames
            destination = destinations[0] # Current destination
            f = cv2.undistort(f, mtx, dist, None, newmtx)
            f = f[:, 200:800]
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            options = apriltag.DetectorOptions(families="tag36h11")
            detector = apriltag.Detector()
            results = detector.detect(gray)
            for detection in results: # Loop over detections
                translation_distance, translation_vector, robot_vector = calculate_required_translation(detection, destination)
                angle = calculate_angle(robot_vector, translation_vector)
                
                # Send angle + distance distance every 10 frames - negative values when the next checkpoint is the block (different arduino control)
                if frame_counter % comm_interval == 0:
                    if len(destinations) == 1:
                        information = str(int(angle)) + ";" + str(round((-translation_distance), 2)) 
                        client.publish("IDP211", information)
                    else:
                        information = str(int(angle)) + ";" + str(round(translation_distance, 2)) 
                        client.publish("IDP211", information)
                
                # Regardless of frame count, send the first distance below the threshold for arduino to break out
                if translation_distance <= 0.10:
                    if len(destinations) == 1:
                        information = str(int(angle)) + ";" + str(round((-translation_distance), 2)) 
                        client.publish("IDP211", information)
                        if abs(angle) < 3: # Extra check when it is the final destination - greater accuracy
                            del destinations[0] # Start listening here
                    else: 
                        information = str(int(angle)) + ";" + str(round((translation_distance), 2)) 
                        client.publish("IDP211", information)
                        del destinations[0] 
    
    # Wait for communications with Arduino - first msg - block colour, second msg - whether pick up was successful
    block_colour = None 
    picked_up = None

    def on_message(client, userdata, msg):
        global block_colour
        if (msg.payload.decode() == 'red') or (msg.payload.decode() == 'blue'):
            block_colour = msg.payload.decode()

    client.subscribe('IDP211')
    client.on_message = on_message

    while block_colour == None: 
        client.loop()
    
    def on_message(client, userdata, msg):
        global picked_up
        picked_up = msg.payload.decode()

    client.subscribe('IDP211')
    client.on_message = on_message
    
    # Rescan for the block and provide reapproach navigation if the signal 'reapproach' is sent
    while True:
        client.loop()
        if picked_up == 'reapproach': 
            picked_up = None
            destination = detect_block(stream) 
            while destination:
                r, f = stream.read()
                frame_counter += 1
                if frame_counter % 2 == 0:
                    f = cv2.undistort(f, mtx, dist, None, newmtx)
                    f = f[:, 200:800]
                    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    options = apriltag.DetectorOptions(families="tag36h11")
                    detector = apriltag.Detector()
                    results = detector.detect(gray)
                    for detection in results:
                        translation_distance, translation_vector, robot_vector = calculate_required_translation(detection, destination)
                        angle = calculate_angle(robot_vector, translation_vector)
            
                        if frame_counter % comm_interval == 0:
                            information = str(int(angle)) + ";" + str(round((-translation_distance), 2)) 
                            client.publish("IDP211", information)
                        
                        # Regardless of frame_count, send the first distance below the threshold for arduino to break out
                        if translation_distance <= 0.10:
                            information = str(int(angle)) + ";" + str(round((-translation_distance), 2)) 
                            client.publish("IDP211", information)
                            if abs(angle) < 3:
                                destination = None

        if picked_up == 'pickedup':
            break

    destinations = [original_forward_path[-2], bottom_ramp, original_forward_path[-3]] # Does not change regardless of colour
    # Set a return path based on colour + previously placed blocks
    if block_colour == 'blue':
        if blue == 1:
            destinations += [bottom_drop_off, intermediates(bottom_drop_off, drop_off_locations[0])[0], drop_off_locations[0]]
            del drop_off_locations[0] 
        elif blue == 0:
            destinations += [top_drop_off, intermediates(top_drop_off, drop_off_locations[0])[0], drop_off_locations[0]]
            del drop_off_locations[0] 
    else: # Red
        if red == 1:
            destinations = [bottom_drop_off, intermediates(bottom_drop_off, drop_off_locations[-1])[0], drop_off_locations[-1]]
            del drop_off_locations[-1]
        elif red == 0:
            destinations = [top_drop_off, intermediates(top_drop_off, drop_off_locations[-1])[0], drop_off_locations[-1]]
            del drop_off_locations[-1]
    
    # Navigate through the drop off path
    while len(destinations) > 0:
        r, f = stream.read()
        frame_counter += 1
        if frame_counter % 2 == 0:
            f = cv2.undistort(f, mtx, dist, None, newmtx)
            f = f[:, 200:800]
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            options = apriltag.DetectorOptions(families="tag36h11")
            detector = apriltag.Detector()
            results = detector.detect(gray)
            destination = destinations[0]
            for detection in results:
                translation_distance, translation_vector, robot_vector = calculate_required_translation(detection, destination)
                angle = calculate_angle(robot_vector, translation_vector)
                
                if frame_counter % comm_interval == 0:
                    if len(destinations) == 1:
                        information = str(int(angle)) + ";" + str(round((-translation_distance), 2)) 
                        client.publish("IDP211", information)
                    else:
                        information = str(int(angle)) + ";" + str(round(translation_distance, 2)) 
                        client.publish("IDP211", information)
                
                # Regardless of frame_count, send the first distance below the threshold for arduino to break out
                if translation_distance <= 0.10 and len(destinations) > 1: ###
                    information = str(int(angle)) + ";" + str(round(translation_distance, 2)) 
                    client.publish("IDP211", information)
                    del destinations[0]
                
                if round(translation_distance, 2) <= 0.02 and len(destinations) == 1:
                    information = str(int(angle)) + ";" + str(round(-translation_distance, 2)) 
                    client.publish("IDP211", information)
                    del destinations[0]

    print('Block done!')

# Return to home
destination = starting_location

while destination:
    if frame_counter % 10 == 0:
        r, f = stream.read()
        f = cv2.undistort(f, mtx, dist, None, newmtx)
        f = f[:, 200:800]
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector()
        results = detector.detect(gray)
        for r in results:
            translation_distance, translation_vector, robot_vector = calculate_required_translation(r, destination)
            angle = calculate_angle(robot_vector, translation_vector)
            
            # Send angle + distance distance every 100 frames                       
            if frame_counter % comm_interval == 0:
                information = str(int(angle)) + ";" + str(round(translation_distance, 2)) 
                client.publish("IDP211", information)
            
            # Regardless of frame_count, send the first distance below the threshold for arduino to break out
            if translation_distance <= 0.05:
                information = str(int(angle)) + ";" + str(round((translation_distance), 2))
                client.publish("IDP211", information)
                destination = None

    frame_counter += 1

print('Competition done!')

while True:
    information = 'stop'
    client.publish("IDP211", information)