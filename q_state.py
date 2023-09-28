import json
import random
import time
import warnings

import gym
import gym_super_mario_bros
from matplotlib import pyplot as plt
import numpy as np
from nes_py.wrappers import JoypadSpace

from detectors import mario_loc_detect, exist_enemy, exist_turtle, exist_left_brick, exist_right_brick, exist_pipe, exist_small_hole


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
env = JoypadSpace(env, CUSTOM_MOVEMENT)


# ----------------------------------------------------------
DELAY = 0.0
ACTION_SIZE = 4

TOTAL_EPISODES = 500  # Total episodes
LEARNING_RATE = 0.8  # Learning rate
MAX_STEPS = 1000  # Max steps per episode
DISCOUNT_RATE = 0.95  # Discounting rate

# Exploration parameters
EPSILON = 1.0  # Exploration rate
DECAY_RATE = 0.005  # Exponential decay rate for exploration prob
# ----------------------------------------------------------

def getState(input_obs) -> str:
    mario_loc = mario_loc_detect(input_obs)
    enemy_loc = exist_enemy(input_obs)
    turtle_loc = exist_turtle(input_obs)
    left_brick_loc = exist_left_brick(mario_loc[0], input_obs)
    right_brick_loc = exist_right_brick(mario_loc[0], input_obs)
    pipe_loc = exist_pipe(input_obs)
    small_hole_loc = exist_small_hole(input_obs)

    result_state = (mario_loc, enemy_loc, turtle_loc, left_brick_loc, right_brick_loc, pipe_loc, small_hole_loc)

    return str(result_state)

def getMaxAction(input_state, input_q_table) -> int:
    max_action = -1
    if (len(input_q_table) == 0):
        print("Q table is empty")
        return 0
    for act in range(ACTION_SIZE):
        curr_pair = tuple((input_state, act))
        if (curr_pair not in input_q_table.keys()):
            continue
        if (input_q_table.get(curr_pair) > max_action):
            max_action = act

    return max_action

def getMaxValue(input_state, input_q_table) -> float:
    max_value = 0
    if (len(input_q_table) == 0):
        print("Q table is empty")
        return 0
    for act in range(ACTION_SIZE):
        curr_pair = tuple((input_state, act))
        if (curr_pair not in input_q_table.keys()):
            continue
        if (input_q_table.get(curr_pair) > max_value):
            max_value = input_q_table.get(curr_pair)

    return max_value

# ----------------------------------------------------------

# List of rewards
rewards = []
# Q-table dictionary initialization
q_table = {}

for episode in range(TOTAL_EPISODES):
    print("\nEpisode: " + str(episode + 1))
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    
    state = getState(obs)

    step = 0
    done = False
    total_rewards = 0

    for step in range(MAX_STEPS):
        explore_exploit_tradeoff = random.uniform(0, 1)

        if explore_exploit_tradeoff > EPSILON and getMaxAction(state, q_table) in range(0, 4):
            action = getMaxAction(state, q_table)
        else:
            action = random.randint(0, 3)

        obs, reward, terminated, truncated, info = env.step(action)
        
        time.sleep(DELAY)

        new_state = getState(obs)

        old_value = q_table.get(tuple((state, action))) or 0

        # Update Q(s,a)= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        pair = tuple((new_state, action))
        if (pair not in q_table.keys() or len(q_table) == 0):
            q_table[pair] = 0
        else:
            q_table[pair] += LEARNING_RATE * (reward + DISCOUNT_RATE * getMaxValue(new_state, q_table) - old_value) # type: ignore

        total_rewards += reward
        state = new_state

        if (terminated or truncated):
            break

    EPSILON = np.exp(-DECAY_RATE * episode)

    rewards.append(total_rewards)
    print("Score: " + str(total_rewards))

print("Score over time: " + str(sum(rewards) / TOTAL_EPISODES))

# ----------------------------------------------------------

# Save q_table to json file
with open("q_table.json", "w", encoding="utf-8") as json_file:
    json.dump(q_table, json_file)


action_list = []

env.reset()
obs, reward, terminated, truncated, info = env.step(0)
state = getState(obs)

for move in range(100):
    action = getMaxAction(state, q_table)
    if (action == -1):
        action = random.randint(0, 3)
    obs, reward, terminated, truncated, info = env.step(action)
    state = getState(obs)
    action_list.append(action)

env.close()

print(action_list)


# Save action_list to txt file
with open("action_list.txt", "w", encoding="utf-8") as f:
    for action in action_list:
        f.write(str(action) + "\n")

# Plot the rewards over episodes
plt.plot(rewards)
plt.show()
