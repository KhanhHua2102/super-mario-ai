from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

import cv2
import numpy as np
import matplotlib.pyplot as plt

import warnings

# Suppress all warnings (not recommended for production code)
warnings.filterwarnings("ignore")

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Initialize the environment and the agent
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, SIMPLE_MOVEMENT)


# Random agent simulation
done = True
env.reset()
for step in range(800):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
env.close()


# Convert the observation to BGR format for cv2 library
obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

# select enemy's head and create a mask
low = np.array([12, 90, 226])
high = np.array([16, 94, 230])
mask = cv2.inRange(obs_img, low, high)

plt.imshow(mask, cmap='gray')
plt.show()

# Find the contours of the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cont_img = cv2.drawContours(obs_img, contours, -1, (0, 255, 0), 2)
plt.imshow(cont_img)
plt.show()

# Area of enemy's head: 105
for contour in contours:
    if cv2.contourArea(contour) < 106 and cv2.contourArea(contour) > 104:
        # Draw a rectangle around the enemy's head
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(obs_img, (x, y), (x + w, y + h), 255, 1)

# Display the original canvas and the result
plt.imshow(obs_img)
plt.show()