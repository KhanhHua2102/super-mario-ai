import json
import random
import time
import warnings

import gym
import gym_super_mario_bros
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from matplotlib import pyplot as plt
import numpy as np
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
# Create the base environment
env = gym.make(
    "SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="rgb_array"
)
env = GrayScaleObservation(env, keep_dim=False)
env = JoypadSpace(env, CUSTOM_MOVEMENT)


DELAY = 0.0
ACTION_SIZE = 4

TOTAL_EPISODES = 50  # Total episodes
LEARNING_RATE = 0.8  # Learning rate
MAX_STEPS = 1000  # Max steps per episode
GAMMA = 0.95  # Discounting rate

# Exploration parameters
EPSILON = 1.0  # Exploration rate
MAX_EPSILON = 1.0  # Exploration probability at start
MIN_EPSILON = 0.01  # Minimum exploration probability
DECAY_RATE = 0.005  # Exponential decay rate for exploration prob

# List of rewards
rewards = []

q_table = {}


def getMaxAction(input_state, input_q_table) -> int:
    max_action = -1
    if (len(input_q_table) == 0):
        return -1
    for i in range(ACTION_SIZE):
        curr_state = (str(input_state) + str(i))
        if (curr_state not in input_q_table.keys()):
            continue
        if (input_q_table.get(curr_state) > max_action):
            max_action = i
        
    return max_action

def getMaxValue(input_state, input_q_table) -> float:
    max_value = 0
    if (len(input_q_table) == 0):
        return -1
    for i in range(ACTION_SIZE):
        curr_state = (str(input_state), str(i))
        if (curr_state not in input_q_table.keys()):
            continue
        if (input_q_table.get(curr_state) > max_value):
            max_value = input_q_table.get(curr_state)

        if (max_value == 0):
            raise Exception("Max value is 0")
    return max_value


for episode in range(TOTAL_EPISODES):
    print("\nEpisode: " + str(episode + 1))
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)

    step = 0
    done = False
    total_rewards = 0

    for step in range(MAX_STEPS):
        exploration_exploitation_tradeoff = random.uniform(0, 1)

        if exploration_exploitation_tradeoff > EPSILON and getMaxAction(obs, q_table) != -1:
            action = getMaxAction(obs, q_table)
        else:
            action = random.randint(0, 3)

        obs, reward, terminated, truncated, info = env.step(action)
        obs = obs[obs.shape[0] // 2 : obs.shape[0] - 15, :]
        time.sleep(DELAY)
        
        done = terminated or truncated

        # Update Q(s,a)= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        newState = (str(obs) + str(action))
        if (newState not in q_table.keys() or len(q_table) == 0):
            q_table[newState] = LEARNING_RATE * reward * getMaxValue(obs, q_table)
        else:
            q_table[newState] = float(q_table.get(newState)) * (1 - LEARNING_RATE) + LEARNING_RATE * (reward + GAMMA * getMaxValue(obs, q_table))

        total_rewards += reward
        state = newState

        if done:
            break

    EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)
    rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards) / TOTAL_EPISODES))

with open("q_table.json", "w", encoding="utf-8") as json_file:
    json.dump(q_table, json_file)


action_list = []

env.reset()
obs, reward, terminated, truncated, info = env.step(0)
obs = obs[obs.shape[0] // 2 : obs.shape[0] - 15, :]

for move in range(100):
    action = getMaxAction(obs, q_table)
    if (action == -1):
        action = random.randint(0, 3)
    obs, reward, terminated, truncated, info = env.step(action)
    obs = obs[obs.shape[0] // 2 : obs.shape[0] - 15, :]
    action_list.append(action)

env.close()

print(action_list)

with open("action_list.txt", "w", encoding="utf-8") as f:
    for action in action_list:
        f.write(str(action) + "\n")
