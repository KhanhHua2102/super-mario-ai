from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

import random
import numpy as np  
import warnings

# Suppress all warnings (not recommended for production code)
warnings.filterwarnings("ignore")

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Create the base environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, SIMPLE_MOVEMENT)


action_size = env.action_space.n
state_size = 12

# action space:
# 0: NOOP
# 1: right
# 2: right + A
# 3: right + B
# 4: right + A + B
# 5: A
# 6: left

# state space:
# 1: small pipe in range
# 2: medium pipe in range
# 3: large pipe in range
# 4: near pipe small pipe
# 5: near pipe medium pipe
# 6: near pipe large pipe
# 7: 1 enemy in range
# 8: 2 enemies in range
# 9: 1 and 2 enemies in range
# 10: 2 and 1 enemies in range
# 11: 3 enemies in range
# 12: hole in range

q_table = np.zeros((state_size, action_size))
print(q_table.shape)

total_episodes = 5000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 5000                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.005             # Exponential decay rate for exploration prob


# List of rewards
rewards = []

for episode in range(total_episodes):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    state = get_state(obs)
    
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        exploration_exploitation_tradeoff = random.uniform(0, 1)

        if exploration_exploitation_tradeoff > epsilon:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + gamma * np.max(q_table[new_state, :]))
        
        total_rewards += reward
        state = new_state

        if done == True:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)
print ("Score over time: " +  str(sum(rewards)/total_episodes))
print(q_table)
