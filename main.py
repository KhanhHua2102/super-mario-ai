from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

from gym.wrappers.gray_scale_observation import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

import cv2
import numpy as np

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Create the base environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
# Wrap env in a DummyVecEnv
env = DummyVecEnv([lambda: env])
# Stack the frames
env = VecFrameStack(env, n_stack=4)

done = True

env.reset()

for _idx in range(3):
    obs, reward, done, info = env.step([4])

    if done:
        state = env.reset()
    
env.close()

print(obs.shape)

frame_array = obs[0, :, :, 0]
print(frame_array.shape)

# Convert observation tensor to a valid image for OpenCV
obs_image = frame_array.astype(np.uint8)
print(obs_image)

cv2.imshow('Input Image', obs_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(obs_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

canvas = np.copy(obs_image)
# Initialize an empty image to draw the results on
result_image = np.zeros_like(canvas)

# Iterate through the detected contours and draw bounding rectangles
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result_image, (x, y), (x + w, y + h), 255, 2)

# Display the original canvas and the result
cv2.imshow('Original Canvas', canvas)
cv2.imshow('Object Detection Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()