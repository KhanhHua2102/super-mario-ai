import warnings

import cv2
import gym
import gym_super_mario_bros
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from PIL import Image

from mario_actions import CUSTOM_MOVEMENT_2

# Suppress all warnings (not recommended for production code)
warnings.filterwarnings("ignore")

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
# Create the base environment
env = gym.make(
    "SuperMarioBros-1-1-v3", apply_api_compatibility=True, render_mode="rgb_array"
)
env = GrayScaleObservation(env, keep_dim=False)
env = JoypadSpace(env, CUSTOM_MOVEMENT_2)

env.reset()
obs, reward, terminated, truncated, info = env.step(0)

print(obs.shape)

obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
cv2.imshow("obs_img", obs_img)
cv2.waitKey(0)

desired_width = 30
desired_height = 32
resized_image = cv2.resize(obs_img, (desired_width, desired_height))
print(resized_image.shape)
cv2.imshow("resized_image", resized_image)
cv2.waitKey(0)


# Load your original image
original_image = Image.fromarray(obs_img)

# Choose the size of the pixelation blocks (larger value = more pixelation)
pixelation_size = 5  # You can adjust this value

# Pixelate the image using the block average technique
pixelated_image = original_image.resize(
    (original_image.width // pixelation_size, original_image.height // pixelation_size),
    resample=Image.BILINEAR
).resize(
    (original_image.width, original_image.height),
    Image.NEAREST
)

pixelated_image_array = np.array(pixelated_image)

print(pixelated_image_array.shape)

plt.imshow(pixelated_image_array)
plt.show()
