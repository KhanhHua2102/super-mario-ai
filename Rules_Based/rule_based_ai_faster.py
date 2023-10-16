import time
import warnings

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

import matplotlib.pyplot as plt
import numpy as np

from Image_Detection.detectors import (
    exist_enemy,
    exist_left_brick,
    exist_pipe,
    exist_right_brick,
    exist_small_hole,
    exist_turtle,
    find_nearest_pipe,
    mario_loc_detect,
)
from utils import Stat

# ------------------------------------------------------------

CUSTOM_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["A"],
    ["right", "B"],
]

# Suppress all warnings
warnings.filterwarnings("ignore")

# Environment setup
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = gym.make(
    "SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, CUSTOM_MOVEMENT)

delay = 0  # Delay between each action
done = False
env.reset()
obs, reward, terminated, truncated, info = env.step(0)
x_mario = mario_loc_detect(obs)[0]
x_mario_list = []

# ------------------------------------------------------------

# Initialize stats
stats = Stat()


def low_jump(env, times, delay):
    """
    Define a set of actions to make mario jump low
    """
    obs, reward, terminated, truncated, info = None, None, None, None, None
    for _ in range(times):
        env.step(2)
        time.sleep(delay)
        # wait for mario to fall
        for _ in range(14):
            env.step(0)
            time.sleep(delay)
        obs, reward, terminated, truncated, info = env.step(0)

    return obs, reward, terminated, truncated, info


def high_jump(env, times, delay):
    """
    Define a set of actions to make mario jump high
    """
    obs, reward, terminated, truncated, info = None, None, None, None, None
    for _ in range(times):
        for _ in range(20):
            env.step(2)
            time.sleep(delay)
        # wait for mario to fall
        for _ in range(17):
            env.step(0)
            time.sleep(delay)
        obs, reward, terminated, truncated, info = env.step(0)

    return obs, reward, terminated, truncated, info


def enemy_react(obs, x_mario, y_mario, env, delay):
    """
    Take action if enemy is in front of mario
    """
    enemy_locs = exist_enemy(obs)
    if len(enemy_locs) > 0:
        loc_list = [loc for loc in enemy_locs if loc[0] > x_mario]
        if len(loc_list) > 0:
            loc = min(loc_list)
            x_enemy = loc[0]
            y_enemy = loc[1]
        else:
            x_enemy, y_enemy = None, None
    else:
        x_enemy, y_enemy = None, None

    if (
        x_enemy is not None
        and (x_enemy - x_mario < 42 and x_enemy - x_mario > 28)
        and y_mario == 79
        and y_enemy == 198
    ):
        print("enemy jump")
        obs, reward, terminated, truncated, info = env.step(2)
        stats.update(reward, x_mario, y_mario)
        time.sleep(delay)
        return True, obs, reward, terminated, truncated, info
    return False, None, None, None, None, None


def turtle_react(obs, x_mario, y_mario, env, delay):
    """
    Take action if turtle is in front of mario
    """
    turle_x, turle_y = exist_turtle(obs)
    if (
        turle_x is not None
        and (turle_x - x_mario < 58 and turle_x - x_mario > 28)
        and y_mario == 79
        and turle_y == 195
    ):
        print("turtle jump")
        obs, reward, terminated, truncated, info = env.step(2)
        stats.update(reward, x_mario, y_mario)
        time.sleep(delay)
        return True, obs, reward, terminated, truncated, info
    return False, None, None, None, None, None


def pipe_react(obs, x_mario, y_mario, env, delay):
    """
    Take action if pipe is in front of mario
    """
    pipe_values = exist_pipe(obs)
    x_pipe, y_pipe = find_nearest_pipe(x_mario, pipe_values)

    if x_pipe is not None:
        if y_pipe == 184:  # short pipe
            if (x_pipe - x_mario < 50 and x_pipe - x_mario > 40) and y_mario == 79:
                print("short pipe jump")
                _, reward, _, _, _ = env.step(3)
                stats.update(reward, x_mario, y_mario)
                time.sleep(delay)
                obs, reward, terminated, truncated, info = env.step(1)
                stats.update(reward, x_mario, y_mario)
                time.sleep(delay)
                return True, obs, reward, terminated, truncated, info
        elif y_pipe == 168:  # medium pipe
            if (x_pipe - x_mario < 65 and x_pipe - x_mario > 27) and y_mario == 79:
                print("medium pipe jump")
                for _ in range(9):
                    _, reward, _, _, _ = env.step(2)
                    stats.update(reward, x_mario, y_mario)
                    time.sleep(delay)
                obs, reward, terminated, truncated, info = env.step(1)
                stats.update(reward, x_mario, y_mario)
                time.sleep(delay)
                return True, obs, reward, terminated, truncated, info
        else:  # long pipe
            if (x_pipe - x_mario < 75 and x_pipe - x_mario > 27) and y_mario == 79:
                print("long pipe jump")
                for _ in range(20):
                    _, reward, _, _, _ = env.step(2)
                    stats.update(reward, x_mario, y_mario)
                    time.sleep(delay)
                obs, reward, terminated, truncated, info = env.step(1)
                stats.update(reward, x_mario, y_mario)
                time.sleep(delay)
                return True, obs, reward, terminated, truncated, info
    return False, None, None, None, None, None


def hole_react(obs, x_mario, y_mario, env, delay):
    """
    Take action if small hole is in front of mario
    """
    small_hole = exist_small_hole(obs)
    if (
        small_hole != (None, None)
        and small_hole[0] - x_mario <= 3
        and small_hole[0] - x_mario > 0
        and y_mario == 79
    ):
        print("hole jump")
        for _ in range(7):
            obs, reward, terminated, truncated, info = env.step(2)
            stats.update(reward, x_mario, y_mario)
            time.sleep(delay)
        return True, obs, reward, terminated, truncated, info
    return False, None, None, None, None, None


def left_brick_react(obs, x_mario, env, delay):
    """
    Take action if left brick is in front of mario
    """
    left_brick = exist_left_brick(x_mario, obs)
    if left_brick != (None, None):
        if left_brick[0] - x_mario <= 50 and left_brick[0] - x_mario > -30:
            obs, reward, terminated, truncated, info = high_jump(env, 2, delay)
            stats.update(reward, x_mario, y_mario)
            time.sleep(delay)
            return True, obs, reward, terminated, truncated, info
    return False, None, None, None, None, None


def right_brick_react(obs, x_mario, env, delay):
    """
    Take action if right brick staircase is in front of mario
    """
    right_brick = exist_right_brick(x_mario, obs)
    if right_brick != (None, None):
        if right_brick[0] - x_mario <= 65 and right_brick[0] - x_mario > -40:
            for i in range(16):
                obs, reward, terminated, truncated, info = env.step(2)
                stats.update(reward, x_mario, y_mario)
                time.sleep(delay)
                return True, obs, reward, terminated, truncated, info
    return False, None, None, None, None, None


def mario_location(obs, x_mario):
    """
    Return mario's location
    """
    x_tmp, _ = mario_loc_detect(obs)
    if x_tmp is not None:
        x_mario = x_tmp
    x_mario_info = info["x_pos"]
    y_mario = info["y_pos"]
    return x_mario_info, x_mario, y_mario


# record start time
start = time.time()
print("\nStart running...\n")
while not done:
    try:
        # Mario's position
        x_mario_info, x_mario, y_mario = mario_location(obs, x_mario)

        # If mario stand still for 8 steps, jump continuously
        x_mario_list.insert(0, x_mario_info)
        steps = 8
        stand_still = False
        if len(x_mario_list) >= steps:
            tmp = x_mario_list[-1]
            for i in range(steps):
                if x_mario_list[i] != tmp:
                    stand_still = False
                    x_mario_list = []
                    break
                stand_still = True
        if stand_still:
            print("\nmario stand still\n")
            high_jump(env, 3, delay)
            time.sleep(delay)

        # Enemy
        results = enemy_react(obs, x_mario, y_mario, env, delay)
        if results[0] is True:
            obs, reward, terminated, truncated, info = results[1:]
            continue

        # Turtle
        results = turtle_react(obs, x_mario, y_mario, env, delay)
        if results[0] is True:
            obs, reward, terminated, truncated, info = results[1:]
            continue

        # Pipe
        results = pipe_react(obs, x_mario, y_mario, env, delay)
        if results[0] is True:
            obs, reward, terminated, truncated, info = results[1:]
            continue

        # Small hole
        results = hole_react(obs, x_mario, y_mario, env, delay)
        if results[0] is True:
            obs, reward, terminated, truncated, info = results[1:]
            continue

        # Left Brick
        results = left_brick_react(obs, x_mario, env, delay)
        if results[0] is True:
            obs, reward, terminated, truncated, info = results[1:]
            continue

        # Right Brick
        results = right_brick_react(obs, x_mario, env, delay)
        if results[0] is True:
            obs, reward, terminated, truncated, info = results[1:]
            continue

        obs, reward, terminated, truncated, info = env.step(4)
        stats.update(reward, x_mario, y_mario)
        time.sleep(delay)

        done = terminated or truncated

    except ValueError:
        print("\nThe End")
        break

env.close()

# ------------------------------------------------------------

# Agent Stats
end = time.time()
print("Time of execution:", round((end - start) * 10**3, 1), "ms")
print("Time left in game: ", info["time"], "ms")
print("Score: ", info["score"])
print("Number of actions: ", stats.get_nums_action())
print("FPS: ", round(stats.get_nums_action() / (end - start), 1))
print("Furthest distance: ", info["x_pos"])
print("Total reward: ", stats.get_total_reward()[-1])
print()

# ------------------------------------------------------------

# Plotting

# Plot total reward over number of action time as line chart
plt.plot(stats.get_total_reward())
plt.title("Total Reward over Number of Action")
plt.xlabel("Number of Action")
plt.ylabel("Total Reward")
plt.show()

# Plot heatmap of agent's activity
level_map = stats.get_heatmap()
plt.imshow(level_map, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.gca().invert_yaxis()
plt.title("Agent's Activity Heatmap")
plt.show()
