import json
import random
import warnings

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from Q_Learning.q_obs import get_max_action

CUSTOM_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["A"],
]

# Suppress all warnings
warnings.filterwarnings("ignore")

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, CUSTOM_MOVEMENT)


DELAY = 0  # Delay between actions
FRAME_SKIP = 4  # Number of frames to skip

# read q_table from file
with open("./Q_Learning/q_learning_model/q_table_obs.json", "r", encoding="utf-8") as f:
    q_table = json.load(f)


action_list = []
env.reset()
obs, reward, terminated, truncated, info = env.step(0)
# Crop the image to bottom half
obs = obs[obs.shape[0] // 2 : obs.shape[0] - 15, :]

for move in range(100):
    action = get_max_action(obs, q_table)
    if action == -1:
        action = random.randint(0, 3)
    for _ in range(FRAME_SKIP - 1):
        env.step(action)
    obs, reward, terminated, truncated, info = env.step(action)
    # Crop the image to bottom half
    obs = obs[obs.shape[0] // 2 : obs.shape[0] - 15, :]
    action_list.append(action)

env.close()

print(action_list)
