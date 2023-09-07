import time
import warnings

import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import mario_actions as ac
from detectors import (
    exist_enemy,
    exist_left_brick,
    exist_pipe,
    exist_right_brick,
    exist_small_hole,
    exist_turtle,
    find_nearest_pipe,
    mario_loc_detect,
)
from mario_actions import CUSTOM_MOVEMENT

# Suppress all warnings (not recommended for production code)
warnings.filterwarnings("ignore")

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = gym.make(
    "SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

delay = 0

done = False
env.reset()
obs, reward, terminated, truncated, info = env.step(0)
x_mario = mario_loc_detect(obs)[0]

x_mario_list = []

# record start time
start = time.time()
print("\nStart running...")
while not done:
    # for step in range(1600): # go to large brick
    # for step in range(1300): # go to left brick
    # for step in range(790): # go to enemy after big hole
    # for step in range(680): # go to big hole
    # for step in range(850): # go to big hole
    try:
        # Mario's position
        x_tmp, y_tmp = mario_loc_detect(obs)
        if x_tmp is not None:
            x_mario = x_tmp
        x_mario_info = info["x_pos"]
        y_mario = info["y_pos"]

        x_mario_list.insert(0, x_mario_info)

        # If mario stand still for _ steps, jump continuously
        steps = 7
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
            ac.high_jump(env, 3, delay)
            time.sleep(delay)

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
            and (x_enemy - x_mario < 32 and x_enemy - x_mario > 28)
            and y_mario == 79
            and y_enemy == 198
        ):
            print("enemy jump")
            obs, reward, terminated, truncated, info = env.step(2)
            time.sleep(delay)
            continue

        # Turtle
        turle_x, turle_y = exist_turtle(obs)
        if (
            turle_x is not None
            and (turle_x - x_mario < 33 and turle_x - x_mario > 28)
            and y_mario == 79
            and turle_y == 195
        ):
            print("turtle jump")
            obs, reward, terminated, truncated, info = env.step(2)
            time.sleep(delay)
            continue

        pipe_values = exist_pipe(obs)
        x_pipe, y_pipe = find_nearest_pipe(x_mario, pipe_values)

        if x_pipe is not None:
            if y_pipe == 184:  # short pipe
                if (x_pipe - x_mario < 45 and x_pipe - x_mario > 40) and y_mario == 79:
                    print("short pipe jump")
                    env.step(5)
                    time.sleep(delay)
                    obs, reward, terminated, truncated, info = env.step(1)
                    time.sleep(delay)
                    continue
            elif y_pipe == 168:  # medium pipe
                if (x_pipe - x_mario < 65 and x_pipe - x_mario > 27) and y_mario == 79:
                    print("medium pipe jump")
                    for _ in range(12):
                        env.step(2)
                        time.sleep(delay)
                    obs, reward, terminated, truncated, info = env.step(1)
                    time.sleep(delay)
                    continue
            else:  # long pipe
                if (x_pipe - x_mario < 75 and x_pipe - x_mario > 27) and y_mario == 79:
                    print("long pipe jump")
                    for _ in range(20):
                        env.step(2)
                        time.sleep(delay)
                    obs, reward, terminated, truncated, info = env.step(1)
                    time.sleep(delay)
                    continue

        # Small hole
        small_hole = exist_small_hole(obs)
        if (
            small_hole != (None, None)
            and small_hole[0] - x_mario <= 3
            and small_hole[0] - x_mario > 0
            and y_mario == 79
        ):
            print("hole jump")
            for _ in range(18):
                obs, reward, terminated, truncated, info = env.step(2)
                time.sleep(delay)

        # Left Brick
        left_brick = exist_left_brick(x_mario, obs)
        if left_brick != (None, None):
            if left_brick[0] - x_mario <= 39 and left_brick[0] - x_mario > -30:
                for _ in range(20):
                    env.step(2)
                    time.sleep(delay)
                for _ in range(19):
                    env.step(0)
                for _ in range(18):
                    env.step(2)
                    time.sleep(delay)
                for _ in range(15):
                    obs, reward, terminated, truncated, info = env.step(0)

        x_tmp, y_tmp = mario_loc_detect(obs)
        if x_tmp is not None:
            x_mario = x_tmp
        y_mario = info["y_pos"]

        # Right Brick
        right_brick = exist_right_brick(x_mario, obs)
        if right_brick != (None, None):
            if right_brick[0] - x_mario <= 65 and right_brick[0] - x_mario > -40:
                for i in range(16):
                    obs, reward, terminated, truncated, info = env.step(2)
                    time.sleep(delay)

        obs, reward, terminated, truncated, info = env.step(1)
        time.sleep(delay)

        done = terminated or truncated

    except ValueError:
        print("\nThe End")
        break

env.close()

end = time.time()
print("Time of execution:", round((end - start) * 10**3, 1), "ms")
print("Time in game: ", info["time"], "ms")
