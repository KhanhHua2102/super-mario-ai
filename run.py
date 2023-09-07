import time
import warnings

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import mario_actions as ac
from mario_states import CUSTOM_MOVEMENT

# Suppress all warnings (not recommended for production code)
warnings.filterwarnings("ignore")

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, CUSTOM_MOVEMENT)

delay = 0.05
# read action list from file
with open("action_list.txt", "r") as f:
    action_list = [int(line.strip()) for line in f.readlines()]

print("action list:", action_list)

delay = 0.1

done = True
env.reset()
for action in action_list:
    print("action:", action)
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)
    
    # for i in range(12):
    #     env.step(0)
    time.sleep(delay)
    done = terminated or truncated

