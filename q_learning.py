import random
import time
import warnings

import gym
import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace

from mario_actions import CUSTOM_MOVEMENT
from states import get_state

# Suppress all warnings (not recommended for production code)
warnings.filterwarnings("ignore")

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
# Create the base environment
env = gym.make(
    "SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="rgb_array"
)
env = JoypadSpace(env, CUSTOM_MOVEMENT)


action_size = 5
state_size = 6

# action space:
# 0: NOOP
# 1: right
# 2: right + A
# 3: A
# 4: right + B

# state space:
# 0: small pipe in range
# 1: medium pipe in range
# 2: large pipe in range
# 3: 1 enemy in range
# 4: hole in range
# 5: nothing in range

q_table = np.zeros((state_size, action_size))
print(q_table.shape)

delay = 0.05

total_episodes = 5000  # Total episodes
learning_rate = 0.8  # Learning rate
max_steps = 5000  # Max steps per episode
gamma = 0.95  # Discounting rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob

# List of rewards
rewards = []

for episode in range(total_episodes):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    state = get_state(obs, info)
    
    print("Episode: " + str(episode))
    print("State: " + str(state))

    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        exploration_exploitation_tradeoff = random.uniform(0, 1)

        if exploration_exploitation_tradeoff > epsilon:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
    
        print("Action: " + str(action))
        
        time.sleep(delay)

        obs, reward, terminated, truncated, info = env.step(action)
        new_state = get_state(obs, info)
        
        time.sleep(delay)
        
        print("Reward: " + str(reward))
        print("New state: " + str(new_state))
        
        done = terminated or truncated

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        q_table[state, action] = q_table[state, action] * (
            1 - learning_rate
        ) + learning_rate * (reward + gamma * np.max(q_table[new_state, :]))

        total_rewards += reward
        state = new_state

        if done == True:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)
print("Score over time: " + str(sum(rewards) / total_episodes))
print(q_table)
