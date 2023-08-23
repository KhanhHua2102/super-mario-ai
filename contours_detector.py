import cv2
import numpy as np
import sys

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
    # print(pipe_values)
    return pipe_values

def find_nearest_pipe(mario_x,mario_y,pipe_values):
    if len(pipe_values) == 0:
        return None,None

    closest_x = pipe_values[0][0]
    closest_y = pipe_values[0][1]

    # closest_x, closest_y = sys.maxsize,sys.maxsize

    #only one pipe exists and the pipe is in front of the mario right now
    if len(pipe_values) == 1 and closest_x > mario_x:
        return closest_x,closest_y

    for pipe_value in pipe_values:  
        # less than other pipes but greater than the mario
        print(pipe_value)
        # 24 168, 152 152
        if closest_x < pipe_value[0] and pipe_value[0] > mario_x and mario_x > closest_x:
            closest_x = pipe_value[0]
            closest_y = pipe_value[1]

    return closest_x,closest_y


def one_enemy(obs):
    return 7