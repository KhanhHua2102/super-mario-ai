import json
import random
import time
import warnings

import gym
import gym_super_mario_bros
import numpy as np
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from matplotlib import pyplot as plt
from nes_py.wrappers import JoypadSpace

from utils import print_stats


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

DELAY = 0.0
ACTION_SIZE = 4

TOTAL_EPISODES = 500  # Total episodes
LEARNING_RATE = 0.8  # Learning rate
MAX_STEPS = 1000  # Max steps per episode
FRAME_SKIP = 4
DISCOUNT_RATE = 0.95  # Discounting rate

# Exploration parameters
EPSILON = 1.0  # Exploration rate
DECAY_RATE = 0.005  # Exponential decay rate for exploration prob

# ----------------------------------------------------------

def get_max_action(input_obs, input_q_table) -> int:
    max_action = -1
    if (len(input_q_table) == 0):
        return -1
    for act in range(ACTION_SIZE):
        curr_pair = tuple((str(input_obs), act))
        if (curr_pair not in input_q_table.keys()):
            continue
        if (input_q_table.get(curr_pair) > max_action):
            max_action = act

    return max_action

def get_max_value(input_obs, input_q_table) -> float:
    max_value = 0
    if (len(input_q_table) == 0):
        return -1
    for act in range(ACTION_SIZE):
        curr_pair = tuple((str(input_obs), act))
        if (curr_pair not in input_q_table.keys()):
            continue
        if (input_q_table.get(curr_pair) > max_value):
            max_value = input_q_table.get(curr_pair)

    return max_value

# ----------------------------------------------------------

# Start timer
start = time.time()
# List of rewards
rewards = []
# Q-table initialization
q_table = {}

for episode in range(TOTAL_EPISODES):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    # Crop the image to bottom half
    state = str(obs[obs.shape[0] // 2 : obs.shape[0] - 15, :])

    step = 0
    done = False
    total_rewards = 0

    for step in range(MAX_STEPS // FRAME_SKIP):
        explore_exploit_tradeoff = random.uniform(0, 1)

        if explore_exploit_tradeoff > EPSILON and get_max_action(obs, q_table) in range(0, 4):
            action = get_max_action(obs, q_table)
        else:
            action = random.randint(0, 3)

        for _ in range(FRAME_SKIP - 1):
            new_obs, reward, terminated, truncated, info = env.step(action)
            if (terminated or truncated):
                break
        if (terminated or truncated):
                break
        obs, reward, terminated, truncated, info = env.step(action) 
        if (terminated or truncated):
            break
        # Crop the image to bottom half
        new_state = obs[obs.shape[0] // 2 : obs.shape[0] - 15, :]
        time.sleep(DELAY)

        old_value = q_table.get(tuple((str(state), action))) or 0

        pair = tuple((str(new_state), action))
        if (len(q_table) == 0 or pair not in q_table.keys()):
            q_table[pair] = 0
        else:
            # Update Q(s,a)= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])
            q_table[pair] +=  LEARNING_RATE * (reward + DISCOUNT_RATE * get_max_value(new_state, q_table) - old_value)

        total_rewards += reward
        state = new_state

    EPSILON = np.exp(-DECAY_RATE * episode)

    rewards.append(total_rewards)
    print_stats(episode, total_rewards, EPSILON, LEARNING_RATE, DISCOUNT_RATE, time.time() - start)

print("Score over time: " + str(sum(rewards) / TOTAL_EPISODES))
print("Training time: " + str(time.time() - start) + " seconds")

# ----------------------------------------------------------

# Save q_table to json file
with open("Q_Learning/q_learning_model/q_table_obs.json", "w", encoding="utf-8") as json_file:
    json.dump(q_table, json_file)


action_list = []

env.reset()
obs, reward, terminated, truncated, info = env.step(0)
# Crop the image to bottom half
obs = obs[obs.shape[0] // 2 : obs.shape[0] - 15, :]

for move in range(100):
    action = get_max_action(obs, q_table)
    if (action == -1):
        action = random.randint(0, 3)
    for _ in range(FRAME_SKIP - 1):
        env.step(action)
    obs, reward, terminated, truncated, info = env.step(action)
    # Crop the image to bottom half
    obs = obs[obs.shape[0] // 2 : obs.shape[0] - 15, :]
    action_list.append(action)

env.close()

print(action_list)


# Save action_list to txt file
with open("Q_Learning/q_learning_model/action_list_obs.txt", "w", encoding="utf-8") as f:
    for action in action_list:
        f.write(str(action) + "\n")

# Plot the rewards over episodes
plt.plot(rewards)
plt.show()