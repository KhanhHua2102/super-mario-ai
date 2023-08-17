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

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, SIMPLE_MOVEMENT)


done = True
env.reset()
for step in range(800):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
env.close()


obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
template = cv2.imread("templates/enemy.png")

obs_img_gray = cv2.cvtColor(obs_img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# cv2.imshow("obs_img", obs_img)
# cv2.imshow("template", template)
# cv2.imshow("obs_img", obs_img_gray)
# cv2.imshow("template", template_gray)

match_method = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]

for method in match_method:
    result = cv2.matchTemplate(obs_img_gray, template_gray, method)
    cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    matchLoc = max_loc
    cv2.rectangle(obs_img, matchLoc, (matchLoc[0] + template.shape[0], matchLoc[1] + template.shape[1]), (0,0,0), 2, 8, 0 )
    cv2.rectangle(result, matchLoc, (matchLoc[0] + template.shape[0], matchLoc[1] + template.shape[1]), (0,0,0), 2, 8, 0 )
    cv2.imshow("image_window", obs_img)
    # cv2.imshow("result_window", result)
 
cv2.waitKey(10000)


min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
h, w = template.shape[:-1]
bottom_right = (top_left[0] + w, top_left[1] + h)

# Draw a bounding box around the matched region
# matched_img = cv2.cvtColor(obs_img.copy(), cv2.COLOR_BGR2RGB)
cv2.rectangle(obs_img, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle

# Print the position of the best match
print("Top left:", top_left)
print("Bottom right:", bottom_right)

# Display the matched image with bounding box using plt.imshow
plt.imshow(obs_img)
plt.show()
