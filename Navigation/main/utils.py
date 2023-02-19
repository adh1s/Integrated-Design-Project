import math
from collections import namedtuple, deque
import numpy as np
import cv2

from cv_utils import detect_block, detect_dropoff, detect_intersections

coordinate = namedtuple("coordinate", "x y")
vector = namedtuple("vector", "x_component y_component")

class Camera():
    """
    Class to store the properties of the camera.
    undistort method used to undistort images from the cv2 stream.
    """
    def __init__(self, mtx: np.array, dist: np.array, newmtx: np.array):
        self.mtx = np.reshape(mtx, (3, 3))
        self.dist = np.reshape(dist, (1,5))
        self.newmtx = np.reshape(newmtx, (3,3))
    def undistort(self, frame):
        return cv2.undistort(frame, self.mtx, self.dist, None, self.newmtx)

class Arena():
    def __init__(self, stream, camera):
        """
        Class to store/scan for the landmarks of the arena; starts with hardcoded values.
        """
        self.starting_location = coordinate(532, 175)
        self.intersection_1 = coordinate(421, 275)
        self.intersection_2 = coordinate(156, 522)
        self.block = coordinate(83, 590)

        self.drop_offs = {'blue': [coordinate(313, 222), coordinate(375, 166)],
                          'red': [coordinate(525, 331), coordinate(467, 389)]} 
        self.blocks = 0 # Blocks delivered
        self.camera = camera
        
        self.scan(stream=stream)

    def scan(self, stream, max_frames=100):
        frame_counter = 0
        while frame_counter < max_frames:
            _, f = stream.read()
            frame_counter += 1
            f = self.camera.undistort(f) 
            f = f[:, 200:800]
            block = detect_block(f)
            intersections = detect_intersections(f)
            drop_offs = detect_dropoff(f)
            if block:
                self.block = block
            if intersections:
                self.intersection_1 = intersections[0]
                self.intersection_2 = intersections[1]
            if drop_offs:
                self.drop_offs['blue'] = [drop_offs[0], drop_offs[1]]
                self.drop_offs['red'] = [drop_offs[2], drop_offs[3]]

    def scan_block(self, stream, max_frames=100):
        frame_counter = 0
        while frame_counter < max_frames:
            _, f = stream.read()
            frame_counter += 1
            f = self.camera.undistort(f) 
            f = f[:, 200:800]
            block = detect_block(f)
            if block:
                self.block = block
                break

class Navigation():
    def __init__(self, arena, camera):
        """
        Class for all things navigation;
        Synthesises paths and follows them (includes helper static methods for navigation)
        """
        self.arena = arena
        self.camera = camera
        self.current_path = deque([]) # Current path to follow 
        self.mode = None # Pick-up, Delivery or Finish

    def forward_path(self):
        """
        Updates current_path with the path required for a pick up
        """
        full_forward_path = deque([self.arena.intersection_1, 
                                    Navigation.intermediates(self.arena.intersection_1, 
                                                            self.arena.intersection_2, nb_points=1), 
                                    self.arena.intersection_2, 
                                    self.arena.block])
        # Does not require the whole forward path if returning from delivery
        if self.arena.blocks != 0:
            full_forward_path.popleft() 
        
        self.current_path = full_forward_path
        self.mode = 'Pick-up'

    def delivery_path(self, colour: str):
        """
        Updates current_path with appropriate delivery path
        """
        delivery_path = deque([self.arena.intersection_2,
                               Navigation.intermediates(self.arena.intersection_1, 
                                                        self.arena.intersection_2, nb_points=1), 
                               self.arena.intersection_1, 
                               self.arena.drop_offs[colour].pop()])
        self.current_path = delivery_path
        self.arena.blocks += 1 
        self.mode = "Delivery"

    def return_path(self):
        """
        Updates current_path with the path required to return home
        """
        home_path = deque([self.arena.starting_location])
        self.current_path = home_path
        self.mode = "Finish"
    
    def reapproach_path(self):
        self.current_path = deque([self.arena.block])

    def update_arena(self, stream):
        """
        Updates the arena with up-to-date scan information
        """
        self.arena.scan_block(stream)
    
    def navigate_path(self, stream, detector, client, topic='IDP211'):
        frame_counter = 0 
        while self.current_path:
            _, f = stream.read()
            frame_counter += 1

            destination = self.current_path[0]
            f = self.camera.undistort(f) 
            f = f[:, 200:800]
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray)

            for detection in results: 
                translation_information = Navigation.calculate_required_translation(detection, 
                                                                              destination)
                translation_distance = translation_information['translation_distance']
                angle = Navigation.calculate_angle(translation_information["robot_vector"],
                                                   translation_information["path_vector"])
                
                # Negative distance for last destination (triggers a slow approach mode)
                if len(self.current_path) == 1:
                    information = str(int(angle)) + ";" + str(round(( - translation_distance), 2)) 
                    client.publish(topic, information)
                else:
                    information = str(int(angle)) + ";" + str(round(translation_distance, 2)) 
                    client.publish(topic, information)
                
                # Delete the checkpoint when below a certain threshold - 'reached'
                if translation_distance <= 0.10:
                    self.current_path.popleft() 

    @staticmethod
    def calculate_required_translation(aruco_tag_detection, destination: coordinate) -> dict:
        """aruco_tag_detection: each result from the detector (data structure in https://pypi.org/project/apriltag/)
           destination: coordinates of the next destination in the trip
        returns {'translation distance': x m, 'path_vector': vector, 'robot_vector': vector} """

        ptA, ptB, _, _ = aruco_tag_detection.corners
        ptB = coordinate(int(ptB[0]), int(ptB[1]))
        ptA = coordinate(int(ptA[0]), int(ptA[1]))

        center = coordinate(int(aruco_tag_detection.center[0]), 
                            int(aruco_tag_detection.center[1]))
        
        # Unit vector in the direction the robot is facing 
        x_comp = ((ptB[0] + ptA[0])/ 2) - center.x
        y_comp = center.y - ((ptB[1] + ptA[1])/2)
        robot_vector = Navigation.normalise(x_comp, y_comp)

        # Unit vector in the direction of required translation 
        translation_x = destination[0] - center.x
        translation_y = center.y - destination[1]
        translation_pixel_distance = (translation_x**2 + translation_y**2)**(1/2)
        path_vector = Navigation.normalise(translation_x, translation_y)

        # Basic pixel to distance calibration
        front_edge = vector(ptB[0] - ptA[0], ptB[1] - ptA[1])
        front_edge_pixel_distance = (front_edge.x_component**2 + front_edge.y_component**2)**(1/2)
        front_edge_length = 0.093 # in m, known distance for calibration reference
        translation_distance = ((translation_pixel_distance / front_edge_pixel_distance) * front_edge_length) 

        return {'translation_distance': translation_distance, 
                'path_vector': path_vector, 
                'robot_vector': robot_vector}

    @staticmethod
    def calculate_angle (rvec: vector, pvec: vector) -> float:
        """rvec: unit vector in the direction the robot is facing
           pvec: unit vector in the direction of required translation
        returns angle (positive - left turn, negative - right turn """
        
        dot = rvec.x*pvec.x + rvec.y*pvec.y   # dot product between [rx, ry] and [px, py]
        det = rvec.x*pvec.y - rvec.y*pvec.x   # determinant
        radians = math.atan2(det, dot)
        return ((180/math.pi)*radians)
    
    @staticmethod
    def intermediates(p1: tuple, p2: tuple, nb_points: int = 1) -> list:
        """p1: first coordinate
           p2: second coordinate
           returns nb_points equally spaced points interpolated with p1 and p2 """

        x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
        y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

        return [(int(p1[0] + i * x_spacing), int(p1[1] +  i * y_spacing))
                for i in range(1, nb_points+1)]
    
    @staticmethod
    def normalise(vec: vector):
        """vec: vector 
        return unit vector in the same direction as vec """
        magnitude = (vec.x_component**2 + vec.y_component**2)**(1/2)
        return vector(vec.x_component/magnitude, vec.y_component/magnitude)