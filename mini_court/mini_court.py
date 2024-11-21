import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt():
    def __init__(self, frame):
        
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        
        self.set_canvas_background_box_position(frame)
        

        
        self.set_mini_court_position()
        

        
        self.set_court_drawing_key_points()
        

        
        self.set_court_lines()
        
    
    def convert_meters_to_pixels(self, meters):
        
        return convert_meters_to_pixel_distance(meters,
                                                constants.PADEL_COURT_WIDTH,
                                                self.court_drawing_width
                                            )
    def set_court_drawing_key_points(self):
        
        drawing_key_points = [0] * 20

        # Point 0: bottom-left corner (court start)
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # Point 1: bottom-right corner (court end)
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # Point 2: top-left corner
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.PADEL_COURT_LENGTH)
        # Point 3: top-right corner
        drawing_key_points[6] = drawing_key_points[2]
        drawing_key_points[7] = drawing_key_points[5]
        #bottom left +3
        drawing_key_points[8] = int(self.court_start_x)
        drawing_key_points[9] = self.court_start_y + self.convert_meters_to_pixels(constants.PADEL_SERVICE_LINE_DISTANCE)
        #bottom right +3
        drawing_key_points[10] =int(self.court_end_x) 
        drawing_key_points[11] = self.court_start_y + self.convert_meters_to_pixels(constants.PADEL_SERVICE_LINE_DISTANCE)
        #middel bottom 
        drawing_key_points[12] =drawing_key_points[0]+ self.convert_meters_to_pixels(constants.PADEL_COURT_WIDTH/2)
        drawing_key_points[13] =drawing_key_points[11]
        #top left -3
        drawing_key_points[14] = int(self.court_start_x)
        drawing_key_points[15] =drawing_key_points[5]-self.convert_meters_to_pixels(constants.PADEL_SERVICE_LINE_DISTANCE)
        #top right -3
        drawing_key_points[16] =int(self.court_end_x)
        drawing_key_points[17] =drawing_key_points[15]
        #middel top
        drawing_key_points[18] =drawing_key_points[12]
        drawing_key_points[19] =drawing_key_points[15]

        self.drawing_key_points = drawing_key_points
        

    def set_court_lines(self):
        
        self.lines = [
         (0, 2),  # left side -> (0,1) (4,5)
            (4, 5),  # top side
            (1, 3),  # bottom side
            (0, 1),  # bottom line
            (7, 8),  # left service line -> (16,17) (18,19)
            (2, 3),  # top line
            (6,9)
        ]
       

    def draw_background_rectangle(self, frame):
       
        shapes = np.zeros_like(frame, np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out
   
    def draw_court(self, frame):
       
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
           
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
          
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
       
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)
        return frame
    
    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames
    
    def set_mini_court_position(self):
        
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        
    def set_canvas_background_box_position(self, frame):
        
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height
    
    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    