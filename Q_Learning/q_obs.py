import json
import os
import random
import time
import warnings

import gym
import gym_super_mario_bros
import numpy as np
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from matplotlib import pyplot as plt
from nes_py.wrappers import JoypadSpace

from utils import print_stats, save_q_table

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
    "SuperMarioBros-1-1-v3", apply_api_compatibility=True, render_mode="rgb_array"
)
env = GrayScaleObservation(env, keep_dim=False)
env = JoypadSpace(env, CUSTOM_MOVEMENT)

# ----------------------------------------------------------

# Environment parameters
DELAY = 0.0
ACTION_SIZE = 4

# Training parameters
TOTAL_EPISODES = 3000
MAX_STEPS = 1000
FRAME_SKIP = 4
GAMMA = 0.95

# Learning parameters
learning_rate = 0.8
MIN_LEARNING_RATE = 0.05
DISCOUNT_RATE = 0.00001

# Exploration parameters
exploration_rate = 1.0
MIN_EXPLORE_RATE = 0.01
DECAY_RATE = 0.0001

# ----------------------------------------------------------


def get_max_action(input_obs, input_q_table) -> int:
    max_action = -1
    if len(input_q_table) == 0:
        return -1
    for act in range(ACTION_SIZE):
        curr_pair = str((str(input_obs), act))
        if curr_pair not in input_q_table.keys():
            continue
        if input_q_table.get(curr_pair) > max_action:
            max_action = act

    return max_action


def get_max_value(input_obs, input_q_table) -> float:
    max_value = 0
    if len(input_q_table) == 0:
        return -1
    for act in range(ACTION_SIZE):
        curr_pair = str((str(input_obs), act))
        if curr_pair not in input_q_table.keys():
            continue
        if input_q_table.get(curr_pair) > max_value:
            max_value = input_q_table.get(curr_pair)

    return max_value


# ----------------------------------------------------------

# Start timer
start = time.time()
# List of rewards
rewards = []
# Q-table initialization
q_table = {}

if os.path.exists("Q_Learning/train/current_model_obs.json") and os.path.exists(
    "Q_Learning/train/model_statistics_obs.json"
):
    with open("Q_Learning/train/model_statistics_obs.json", "r") as f:
        conf = json.load(f)
        if conf["iterations"] > 0:
            start_episode = conf["iterations"]
    # This is the AI model started
    with open("Q_Learning/train/current_model_obs.json", "r") as f:
        q_table = json.load(f)

for episode in range(start_episode, TOTAL_EPISODES):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    # Crop the image to bottom half
    state = str(obs[obs.shape[0] // 2 : obs.shape[0] - 15, :])

    step = 0
    done = False
    total_rewards = 0

    for step in range(MAX_STEPS // FRAME_SKIP):
        explore_exploit_tradeoff = random.uniform(0, 1)

        if explore_exploit_tradeoff > exploration_rate and get_max_action(
            obs, q_table
        ) in range(0, 4):
            action = get_max_action(obs, q_table)
        else:
            action = random.randint(0, 3)

        for _ in range(FRAME_SKIP - 1):
            new_obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        if terminated or truncated:
            break
        obs, reward, terminated, truncated, info = env.step(action)

        # Crop the image to bottom half
        new_state = obs[obs.shape[0] // 2 : obs.shape[0] - 15, :]
        time.sleep(DELAY)

        old_value = q_table.get(str((str(state), action))) or 0

        pair = str((str(new_state), action))
        if len(q_table) == 0 or pair not in q_table.keys():
            q_table[pair] = 0
        else:
            # Update Q(s,a)= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])
            q_table[pair] += learning_rate * (
                reward + GAMMA * get_max_value(new_state, q_table) - old_value
            )

        total_rewards += reward
        state = new_state

        if terminated or truncated:
            break

    # Reduce exploration rate (epsilon) and learning rate (alpha) over time
    exploration_rate = max(MIN_EXPLORE_RATE, np.exp(-DECAY_RATE * episode))
    learning_rate = max(MIN_LEARNING_RATE, np.exp(-DISCOUNT_RATE * episode))

    print_stats(
        episode, total_rewards, time.time() - start, exploration_rate, learning_rate
    )

print("Score over time: " + str(sum(rewards) / TOTAL_EPISODES))
print("Training time: " + str(time.time() - start) + " seconds")

# Plot the rewards over episodes
plt.plot(rewards)
plt.show()

# ----------------------------------------------------------

# Save q_table to json file
save_q_table(q_table, episode, "_obs")
