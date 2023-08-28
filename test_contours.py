from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

import cv2
import time

import warnings
import numpy as np

from contours_detector import mario_loc, exist_enemy, exist_pipe,find_nearest_pipe,exist_small_hole,exist_brick

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

delay = 0.0001

done = True
env.reset()
obs, reward, terminated, truncated, info = env.step(0)
for step in range(500):
    # Mario's position
    x, _ = mario_loc(obs)
    y = info["y_pos"]
    print(f"Mario ({x} | {y})")

    x_enemy, y_enemy = exist_enemy(obs)
    if x_enemy is not None:
        # print(f"enemy ({x_enemy} | {y_enemy})")
        if (x_enemy - x < 32 and x_enemy - x > 28) and y <= 79:
            print(f"Mario ({x} | {y})")
            # print(f"enemy ({x_enemy} | {y_enemy})")
            obs, reward, terminated, truncated, info = env.step(2)
            time.sleep(delay)
            continue
    

    pipe_values = exist_pipe(obs)
    x_pipe, y_pipe = find_nearest_pipe(x,pipe_values)


    if x_pipe is not None:
        print(f"pipe ({x_pipe} | {y_pipe})")
        if y_pipe == 184: # short pipe
            print("short pipe")
            if (x_pipe - x < 45 and x_pipe - x > 40) and y <= 79:
                print(f"Mario ({x} | {y})")
                print(f"pipe ({x_pipe} | {y_pipe})")
                env.step(5)
                time.sleep(delay)
                obs, reward, terminated, truncated, info = env.step(1)
                time.sleep(delay)
                continue
        elif y_pipe == 168: # medium pipe
            print("medium pipe")
            print(f"pipe ({x_pipe} | {y_pipe})")
            print(f"Mario ({x} | {y})")
            if (x_pipe - x < 65 and x_pipe - x > 27) and y <= 79:
                # print(f"Mario ({x} | {y})")
                # print(f"pipe ({x_pipe} | {y_pipe})")
                for _ in range(50):
                    env.step(2)
                    time.sleep(delay)
                obs, reward, terminated, truncated, info = env.step(1)
                time.sleep(delay)
                continue
        else: # long pipe
            print("long pipe")
            print(f"pipe ({x_pipe} | {y_pipe})")
            if (x_pipe - x < 75 and x_pipe - x > 27) and y <= 79:
                print(f"Mario ({x} | {y})")
                print(f"pipe ({x_pipe} | {y_pipe})")
                for _ in range(20):
                    obs, reward, terminated, truncated, info = env.step(2)
                    time.sleep(delay)
                obs, reward, terminated, truncated, info = env.step(1)
                time.sleep(delay)
                continue

    # obs, reward, terminated, truncated, info = env.step(2)


    small_hole = exist_small_hole(obs)

    if small_hole[0] - x < 3:
        for i in range(12):
            obs, reward, terminated, truncated, info = env.step(2)
            # obs, reward, terminated, truncated, info = env.step(2)
            time.sleep(delay)

    brick = exist_brick(x,obs)

    if brick is not None:
        if brick[0] - x < 10:
            print("brick at:",brick)
            for i in range(10):
                obs, reward, terminated, truncated, info = env.step(2)
                obs, reward, terminated, truncated, info = env.step(2)
                time.sleep(delay)
   

    obs, reward, terminated, truncated, info = env.step(1)
    time.sleep(delay)


    done = terminated or truncated
    
env.close()



# obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
# # Read the template
# template = cv2.imread("templates/brick1.png")

# # Loop through all the matching methods
# match_method = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
# for method in match_method:
#     result = cv2.matchTemplate(obs_img, template, method)
#     cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )

#     # Find the location of the best match
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#     if (method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED):
#         matchLoc = min_loc
#     else:
#         matchLoc = max_loc

#     pt1 = matchLoc
#     pt2 = matchLoc[0] + template.shape[0], matchLoc[1] + template.shape[1]
#     area = (pt1[0] - pt2[0]) * (pt1[1] - pt2[1])

#     print(f"pt1: {pt1}, pt2: {pt2}\nArea: {area}\n")

#     # Filter the results and draw the rectangle
#     cv2.rectangle(obs_img, pt1, pt2, (0, 255, 0), 1)

# cv2.imshow("image_window", obs_img)
# cv2.waitKey(30000)

