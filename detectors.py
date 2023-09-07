import cv2
import numpy as np


def mario_loc_detect(obs) -> tuple:
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


def exist_enemy(obs) -> list[tuple]:
    enemy_locs = []
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

            enemy_locs.append((center_x, center_y))

    return enemy_locs


def exist_turtle(obs) -> tuple:
    # Convert the observation to BGR format for cv2 library
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    # select enemy's head using lower and upper bounds then create a mask
    low = np.array([67, 159, 251])
    high = np.array([69, 161, 253])
    mask = cv2.inRange(obs_img, low, high)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area of turtle: 22.5
    for contour in contours:
        if cv2.contourArea(contour) <= 22.5 and cv2.contourArea(contour) >= 22.5:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            return center_x, center_y

    return None, None


def exist_pipe(obs) -> list[tuple]:
    # Convert the observation to BGR format for cv2 library
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    # select the pipe using lower and upper bounds then create a mask
    low = np.array([0, 168, 0])
    high = np.array([1, 168, 10])
    mask = cv2.inRange(obs_img, low, high)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pipe_values = []
    center_x, center_y = 0, 0
    # Area of pipe: 137.5
    for contour in contours:
        if cv2.contourArea(contour) < 140 and cv2.contourArea(contour) > 135:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            pipe_values.append((center_x, center_y))
    return pipe_values


def find_nearest_pipe(mario_x, pipe_values) -> tuple:
    if len(pipe_values) == 0:
        return None, None

    closest_x = pipe_values[0][0]
    closest_y = pipe_values[0][1]

    # only one pipe exists and the pipe is in front of the mario right now
    if len(pipe_values) == 1 and closest_x > mario_x:
        return closest_x, closest_y

    for pipe_value in pipe_values:
        # less than other pipes but greater than the mario
        # 24 168, 152 152
        if (
            closest_x < pipe_value[0]
            and pipe_value[0] > mario_x
            and mario_x > closest_x
        ):
            closest_x = pipe_value[0]
            closest_y = pipe_value[1]

    return closest_x, closest_y


def exist_small_hole(obs) -> tuple:
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    # Read the template
    template = cv2.imread("templates/hole.png")

    # Loop through all the matching methods
    # match_method = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]

    result = cv2.matchTemplate(obs_img, template, cv2.TM_CCOEFF)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    matchLoc = max_loc

    # pt1 = matchLoc
    # pt2 = matchLoc[0] + template.shape[0], matchLoc[1] + template.shape[1]

    if matchLoc[1] > 189:
        return matchLoc[0], matchLoc[1]
    return None, None


def exist_left_brick(mario_x, obs):
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    # Read the template
    template = cv2.imread("templates/left_brick.png")
    method = cv2.TM_CCOEFF

    result = cv2.matchTemplate(obs_img, template, method)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    matchLoc = max_loc

    if matchLoc != None and matchLoc == (158, 141):
        return matchLoc[0], matchLoc[1]
    return None, None


def exist_right_brick(mario_x, obs):
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    # Read the template
    template = cv2.imread("templates/right_brick.png")
    method = cv2.TM_CCOEFF

    result = cv2.matchTemplate(obs_img, template, method)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    matchLoc = max_loc

    if matchLoc != None and (
        matchLoc[1] == 142 or matchLoc[1] == 45 or matchLoc[1] == 46
    ):
        return matchLoc[0], matchLoc[1]
    return None, None
