from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

import time

import warnings

from contours_detector import mario_loc, exist_enemy, exist_turtle, exist_pipe,find_nearest_pipe,exist_small_hole, exist_brick

import cv2
from matplotlib import pyplot as plt
import numpy as np

CUSTOM_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['A']
]

# Suppress all warnings (not recommended for production code)
warnings.filterwarnings("ignore")

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

delay = 0.00

done = False
env.reset()
obs, reward, terminated, truncated, info = env.step(0)
x_mario = mario_loc(obs)[0]
# while (not done):
for step in range(990): # go to enemy after big hole
# for step in range(790): # go to enemy after big hole
# for step in range(680): # go to big hole
# for step in range(850): # go to big hole
    # Mario's position
    x_tmp, y_tmp = mario_loc(obs)
    if x_tmp is not None:
        x_mario = x_tmp
    y_mario = info["y_pos"]

    enemy_locs = exist_enemy(obs)
    if len(enemy_locs) > 0:
        loc_list = [loc for loc in enemy_locs if loc[0] > x_mario]
        if len(loc_list) > 0:
            loc = min(loc_list)
            x_enemy = loc[0]
            y_enemy = loc[1]
        else:
            x_enemy, y_enemy = None, None
    else:
        x_enemy, y_enemy = None, None

    if x_enemy is not None and (x_enemy - x_mario < 32 and x_enemy - x_mario > 28) and y_mario == 79 and y_enemy == 198:
        print("enemy jump")
        obs, reward, terminated, truncated, info = env.step(2)
        time.sleep(delay)
        continue

    # Turtle
    turle_x, turle_y = exist_turtle(obs)
    if turle_x is not None and (turle_x - x_mario < 33 and turle_x - x_mario > 28) and y_mario == 79 and turle_y == 195:
        print("turtle jump")
        obs, reward, terminated, truncated, info = env.step(2)
        time.sleep(delay)
        continue

    pipe_values = exist_pipe(obs)
    x_pipe, y_pipe = find_nearest_pipe(x_mario,pipe_values)

    if x_pipe is not None:
        # print(f"pipe ({x_pipe} | {y_pipe})")
        if y_pipe == 184: # short pipe
            if (x_pipe - x_mario < 45 and x_pipe - x_mario > 40) and y_mario == 79:
                print("short pipe jump")
                # print(f"Mario ({x} | {y})")
                # print(f"pipe ({x_pipe} | {y_pipe})")
                env.step(5)
                time.sleep(delay)
                obs, reward, terminated, truncated, info = env.step(1)
                time.sleep(delay)
                continue
        elif y_pipe == 168: # medium pipe
            if (x_pipe - x_mario < 65 and x_pipe - x_mario > 27) and y_mario == 79:
                print("medium pipe jump")
                for _ in range(12):
                    env.step(2)
                    time.sleep(delay)
                obs, reward, terminated, truncated, info = env.step(1)
                time.sleep(delay)
                continue
        else: # long pipe
            if (x_pipe - x_mario < 75 and x_pipe - x_mario > 27) and y_mario == 79:
                print("long pipe jump")
                for _ in range(20):
                    obs, reward, terminated, truncated, info = env.step(2)
                    time.sleep(delay)
                obs, reward, terminated, truncated, info = env.step(1)
                time.sleep(delay)
                continue

    # Small hole
    small_hole = exist_small_hole(obs)
    # print(small_hole)
    # print(x_mario)
    if small_hole != (None, None) and small_hole[0] - x_mario <= 3 and small_hole[0] - x_mario > 0 and y_mario == 79:
        print("hole jump")
        for i in range(18):
            obs, reward, terminated, truncated, info = env.step(2)
            time.sleep(delay)

    # Brick
    brick = exist_brick(x_mario, obs)
    if brick is not None:
        if brick[0] - x_mario < 10:
            print("brick jump")
            print("brick at:", brick)
            for i in range(20):
                obs, reward, terminated, truncated, info = env.step(2)
                time.sleep(delay)
   

    obs, reward, terminated, truncated, info = env.step(1)
    time.sleep(delay)


    done = terminated or truncated
    
env.close()


# obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
# # Read the template
# left_brick_tmplt = cv2.imread("templates/left_brick.png")
# right_brick_tmplt = cv2.imread("templates/right_brick.png")

# # Loop through all the matching methods
# match_method = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
# for method in match_method:
#     result = cv2.matchTemplate(obs_img, left_brick_tmplt, method)
#     cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1 )

#     # Find the location of the best match
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#     if (method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED):
#         matchLoc = min_loc
#     else:
#         matchLoc = max_loc

#     pt1 = matchLoc
#     pt2 = matchLoc[0] + left_brick_tmplt.shape[0], matchLoc[1] + left_brick_tmplt.shape[1]
#     area = (pt1[0] - pt2[0]) * (pt1[1] - pt2[1])

#     print(f"pt1: {pt1}, pt2: {pt2}\nArea: {area}\n")

#     # Filter the results and draw the rectangle
#     if(area < 5000 and area > 4000):
#         cv2.rectangle(obs_img, pt1, pt2, (0, 255, 0), 1)

# cv2.imshow("image_window", obs_img)
# cv2.waitKey(15000)

# plt.imshow(obs)
# plt.axis("off")
# plt.savefig("templates/frame.png", bbox_inches='tight', pad_inches=0)