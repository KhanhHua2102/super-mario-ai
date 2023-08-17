from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

import cv2
import numpy as np

import warnings


def exist_enemy(obs):
    obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    # select enemy's head BGR
    low = np.array([12, 90, 226])
    high = np.array([16, 94, 230])
    mask = cv2.inRange(obs_img, low, high)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area of enemy's head: 105
    for contour in contours:
        if cv2.contourArea(contour) < 106 and cv2.contourArea(contour) > 104:
            x = contour[0][0][0]
            y = contour[0][0][1]
            return x, y
    
    return None, None


# Suppress all warnings (not recommended for production code)
warnings.filterwarnings("ignore")

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, SIMPLE_MOVEMENT)


done = True
env.reset()
for step in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(1)
    
    x, y = exist_enemy(obs)
    if x is not None:
        print(f"enemy detected ({x} | {y})")
        print("Mario at", info["x_pos"], info["y_pos"])

    done = terminated or truncated
    
env.close()