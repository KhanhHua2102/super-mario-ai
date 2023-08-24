import cv2
import numpy as np

def mario_loc(obs):
    # Convert the observation to BGR format for cv2 library
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    # select Mario's belly using lower and upper bounds then create a mask
    low = np.array([0,52,245])
    high = np.array([2,57,250])
    mask = cv2.inRange(obs_img, low, high)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area of 
    for contour in contours:
        if cv2.contourArea(contour) < 250 and cv2.contourArea(contour) > 0:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            return center_x, center_y
    
    return None, None


def exist_enemy(obs):
    # Convert the observation to BGR format for cv2 library
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    # select enemy's head using lower and upper bounds then create a mask
    low = np.array([0, 75, 226])
    high = np.array([16, 100, 230])
    mask = cv2.inRange(obs_img, low, high)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area of enemy's head: 105
    for contour in contours:
        # if cv2.contourArea(contour) < 1 and cv2.contourArea(contour) > 0:
        print("contour",contour[0][0][1])
        if contour[0][-1][1] > 200:
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

    # Area of pipe: 137.5
    for contour in contours:
        if cv2.contourArea(contour) < 140 and cv2.contourArea(contour) > 135:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            return center_x, center_y
    
    return None, None


def exist_hole(obs):
    pass