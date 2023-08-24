import cv2
import numpy as np
import time

def mario_loc(obs):
    # Convert the observation to BGR format for cv2 library
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    # select Mario's belly using lower and upper bounds then create a mask
    low = np.array([0, 54, 248])
    high = np.array([0, 56, 248])
    mask = cv2.inRange(obs_img, low, high)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Area of 
    for contour in contours:
        if cv2.contourArea(contour) < 33 and cv2.contourArea(contour) > 0:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            return center_x, center_y
    
    return None, None


def exist_enemy(obs):
    # Convert the observation to BGR format for cv2 library
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    # select enemy's head using lower and upper bounds then create a mask
    low = np.array([12, 90, 226])
    high = np.array([16, 94, 230])
    mask = cv2.inRange(obs_img, low, high)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area of enemy's head: 105
    for contour in contours:
        if cv2.contourArea(contour) < 106 and cv2.contourArea(contour) > 104:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            return center_x, center_y
    
    return None, None


def exist_pipe(obs):
    # Convert the observation to BGR format for cv2 library
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    # select the pipe using lower and upper bounds then create a mask
    low = np.array([0, 168, 0])
    high = np.array([1, 168, 10])
    mask = cv2.inRange(obs_img, low, high)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pipe_values = []
    center_x,center_y = 0,0
    # Area of pipe: 137.5
    for contour in contours:
        if cv2.contourArea(contour) < 140 and cv2.contourArea(contour) > 135:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            pipe_values.append((center_x,center_y))
    return pipe_values

def find_nearest_pipe(mario_x,pipe_values):
    if len(pipe_values) == 0:
        return None,None

    closest_x = pipe_values[0][0]
    closest_y = pipe_values[0][1]

    #only one pipe exists and the pipe is in front of the mario right now
    if len(pipe_values) == 1 and closest_x > mario_x:
        return closest_x,closest_y

    for pipe_value in pipe_values:  
        # less than other pipes but greater than the mario
        # 24 168, 152 152
        if closest_x < pipe_value[0] and pipe_value[0] > mario_x and mario_x > closest_x:
            closest_x = pipe_value[0]
            closest_y = pipe_value[1]

    return closest_x,closest_y




def exist_hole(mario_x,mario_y,obs):
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    low = np.array([252,136,104])
    high = np.array([252,136,104])
    mask = cv2.inRange(obs_img, low, high)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    possible_holes = []

    for contour in contours:
        for points in contour:
            for point in points:
                x_value = point[0]
                y_value = point[1]
                if x_value > mario_x and y_value < mario_y and y_value != 0:
                    # time.sleep(1)
                    print("possible hole",x_value,y_value)
                    possible_holes.append((x_value,y_value))

    # time.sleep()
    # print("possible_holes",possible_holes)
    if possible_holes:
        # time.sleep(2)
        min_x = min(possible_holes, key=lambda hole: hole[0])[0]
        max_x = max(possible_holes, key=lambda hole: hole[0])[0]
        width = max_x-min_x
        return width
    else:
        return None
    
def exist_small_hole(obs):
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    # Read the template
    template = cv2.imread("templates/hole8.png")

    # Loop through all the matching methods
    match_method = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
    for method in match_method:
        result = cv2.matchTemplate(obs_img, template, method)
        cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )

        # Find the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if (method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED):
            matchLoc = min_loc
        else:
            matchLoc = max_loc
        print(matchLoc)

        # pt1 = matchLoc
        # pt2 = matchLoc[0] + template.shape[0], matchLoc[1] + template.shape[1]
        # area = (pt1[0] - pt2[0]) * (pt1[1] - pt2[1])
        return matchLoc
        # if pt1 is 


    

