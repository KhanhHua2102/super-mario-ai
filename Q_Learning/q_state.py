import random
import time
import warnings

import gym
import gym_super_mario_bros
import numpy as np
from matplotlib import pyplot as plt
from nes_py.wrappers import JoypadSpace

from Image_Detection.detectors import (
    exist_enemy,
    exist_left_brick,
    exist_pipe,
    exist_right_brick,
    exist_small_hole,
    exist_turtle,
    mario_loc_detect,
)
from utils import print_stats, save_q_table

import cProfile, pstats


CUSTOM_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["A"],
]

# Environment parameters
ACTION_SIZE = CUSTOM_MOVEMENT.__len__()

# Suppress all warnings
warnings.filterwarnings("ignore")

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = gym.make(
    "SuperMarioBros-1-1-v3", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, CUSTOM_MOVEMENT)

# ----------------------------------------------------------


def get_state(input_obs) -> str:
    """
    Return the predefined states based on environmental observations.
    """
    mario_loc = mario_loc_detect(input_obs)
    enemy_loc = exist_enemy(input_obs)
    turtle_loc = exist_turtle(input_obs)
    left_brick_loc = exist_left_brick(mario_loc[0], input_obs)
    right_brick_loc = exist_right_brick(mario_loc[0], input_obs)
    pipe_loc = exist_pipe(input_obs)
    small_hole_loc = exist_small_hole(input_obs)

    result_state = (
        mario_loc,
        enemy_loc,
        turtle_loc,
        left_brick_loc,
        right_brick_loc,
        pipe_loc,
        small_hole_loc,
    )

    return str(result_state)


def get_max_action(input_state, input_q_table) -> int:
    """
    Get the action with the highest Q value for the given state.
    """
    max_action = -1
    if len(input_q_table) == 0:
        print("Q table is empty")
        return 0
    for act in range(ACTION_SIZE):
        curr_pair = str(input_state), str(act)
        if curr_pair not in input_q_table.keys():
            continue
        if input_q_table.get(curr_pair) > max_action:
            max_action = act

    return max_action


def get_max_value(input_state, input_q_table) -> float:
    """
    Get the highest Q value for the given state.
    """
    max_value = 0
    if len(input_q_table) == 0:
        print("Q table is empty")
        return 0
    for act in range(ACTION_SIZE):
        curr_pair = str(input_state), str(act)
        if curr_pair not in input_q_table.keys():
            continue
        if input_q_table.get(curr_pair) > max_value:
            max_value = input_q_table.get(curr_pair)

    return max_value


# ----------------------------------------------------------


def main():
    # Training parameters
    TOTAL_EPISODES = 2
    MAX_STEPS = 2000
    FRAME_SKIP = 5
    GAMMA = 0.95  # Discount factor

    # Learning parameters
    learning_rate = 0.8
    MIN_LEARNING_RATE = 0.05
    DISCOUNT_RATE = 0.001

    # Exploration parameters
    exploration_rate = 1.0
    MIN_EXPLORE_RATE = 0.01
    DECAY_RATE = 0.001

    # Start timer
    start = time.time()
    # List of rewards
    rewards = []
    # Q-table dictionary initialization
    q_table = {}

    start_episode = 0

    for episode in range(start_episode, TOTAL_EPISODES):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        state = get_state(obs)

        step = 0
        done = False
        total_rewards = 0

        for step in range(MAX_STEPS // FRAME_SKIP):
            explore_exploit_tradeoff = random.uniform(0, 1)

            if explore_exploit_tradeoff > exploration_rate and get_max_action(
                state, q_table
            ) in range(0, 4):
                action = get_max_action(state, q_table)
            else:
                action = random.randint(0, 3)

            for _ in range(FRAME_SKIP):
                new_obs, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            if terminated or truncated:
                break

            new_state = get_state(new_obs)

            pair = str(state), str(action)
            old_value = q_table.get(pair) or 0

            # Update Q(s,a)= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            if pair not in q_table.keys() or len(q_table) == 0:
                q_table[pair] = 0
            else:
                q_table[pair] += learning_rate * (reward + GAMMA * get_max_value(new_state, q_table) - old_value)  # type: ignore

            total_rewards += reward
            state = new_state

            if terminated or truncated:
                break

        # Reduce exploration rate (epsilon) and learning rate (alpha) over time
        exploration_rate = max(MIN_EXPLORE_RATE, np.exp(-DECAY_RATE * episode))
        learning_rate = max(MIN_LEARNING_RATE, np.exp(-DISCOUNT_RATE * episode))

        rewards.append(total_rewards)
        print_stats(
            episode, total_rewards, time.time() - start, exploration_rate, learning_rate
        )

    print("End Training!")

    # Save q_table to json file
    save_q_table(q_table, TOTAL_EPISODES, learning_rate, exploration_rate, "state")
    print("Score over time: " + str(sum(rewards) / TOTAL_EPISODES))
    print("Training time: " + str(time.time() - start) + " seconds")

    # Plot the rewards over episodes
    plt.plot(rewards)
    plt.show()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.strip_dirs()
    stats.print_stats()

    # Export profiler output to file
    stats = pstats.Stats(profiler)
    stats.dump_stats("Q_Learning/debug/cProfiler")
