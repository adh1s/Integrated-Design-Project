import apriltag
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from utils import *

blocks_to_attempt = 4 # Number of blocks to attempt
comm_interval = 10 # Number of frames between Arduino comms

mqttBroker = "broker.hivemq.com"
client = mqtt.Client("Python")
client.connect(mqttBroker) 
stream = cv2.VideoCapture('http://localhost:8081/stream/video.mjpeg')

# Load camera properties 
mtx = np.fromfile('distortion_correction/cameramatrix.dat', dtype=float)
dist = np.fromfile('distortion_correction/distortionmatrix.dat')
newmtx = np.fromfile('distortion_correction/newcameramatrix.dat')

camera = Camera(mtx=mtx, dist=dist, newmtx=newmtx)

# QR Code detector
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector()

# Detect landmarks of the arena 
arena = Arena(stream=stream, camera=camera)

# Main class used for navigaton
navigator = Navigation(arena=arena, camera=camera)

while (navigator.arena.blocks) < blocks_to_attempt: # loop for each delivery
    navigator.update_arena(stream=stream) # scan at the start of each pick-up

    navigator.forward_path() # synthesise a forward path
    navigator.navigate_path(stream=stream, 
                            detector=detector, 
                            client=client)
     
    # First message - detected block colour ('red' or 'blue')
    # Second message - pick-up status ('pickup' or 're-approach')

    block_colour = None 
    picked_up = None

    def on_message(client, userdata, msg):
        global block_colour
        if (msg.payload.decode() == 'red') or (msg.payload.decode() == 'blue'):
            block_colour = msg.payload.decode()

    client.subscribe('IDP211')
    client.on_message = on_message

    while not block_colour: 
        client.loop()
    
    def on_message(client, userdata, msg):
        global picked_up
        picked_up = msg.payload.decode()

    client.subscribe('IDP211')
    client.on_message = on_message

    while not picked_up:
        client.loop()
        if picked_up == 'pickedup':
            break
        if picked_up == 'reapproach': 
            picked_up = None
            navigator.update_arena(stream=stream) 
            navigator.reapproach_path()
            navigator.navigate_path(stream=stream, detector=detector, 
                                    client=client, topic='IDP211') 

    print("{} block picked up! Delivery coming up.".format(block_colour))

    navigator.delivery_path(colour=block_colour) 
    navigator.navigate_path(stream=stream, detector=detector, 
                            client=client, topic='IDP211') 
    print('Block done!')

# Return home!
navigator.return_path()
navigator.navigate_path(stream=stream, detector=detector, 
                        client=client, topic='IDP211') 
print('Competition done!')