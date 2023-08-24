from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

import cv2
import numpy as np
import matplotlib.pyplot as plt

import warnings

# Suppress all warnings (not recommended for production code)
warnings.filterwarnings("ignore")

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Initialize the environment and the agent
env = gym.make('SuperMarioBros-v3', apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Random agent simulation
done = True
env.reset()
for step in range(1200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
env.close()

# Convert the observation to BGR format for cv2 library
obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
# Read the template
template = cv2.imread("templates/enemy-v3-4.png")

# obs_img_gray = cv2.cvtColor(obs_img, cv2.COLOR_BGR2GRAY)
# template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

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

    pt1 = matchLoc
    pt2 = matchLoc[0] + template.shape[0], matchLoc[1] + template.shape[1]
    area = (pt1[0] - pt2[0]) * (pt1[1] - pt2[1])

    print(f"pt1: {pt1}, pt2: {pt2}\nArea: {area}\n")

    # Filter the results and draw the rectangle
    if(area < 3000 and area > 0):
        cv2.rectangle(obs_img, pt1, pt2, (0, 255, 0), 1)

cv2.imshow("image_window", obs_img)
cv2.waitKey(15000)
