import time
import warnings

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

CUSTOM_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["A"],
]

# Suppress all warnings (not recommended for production code)
warnings.filterwarnings("ignore")
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = gym.make(
    "SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, CUSTOM_MOVEMENT)


DELAY = 0.01
FRAME_SKIP = 0

# read action list from file
with open("Monte_Carlo/action_list.txt", "r", encoding="utf-8") as f:
    action_list = [int(line.strip()) for line in f.readlines()]


env.reset()
for action in action_list:
    print("action:", action)

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

    time.sleep(DELAY)


env.close()
