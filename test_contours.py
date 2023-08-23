from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

import cv2
import time

import warnings

from contours_detector import mario_loc, exist_enemy, exist_pipe,find_nearest_pipe


# Suppress all warnings (not recommended for production code)
warnings.filterwarnings("ignore")

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

delay = 0.008

done = True
env.reset()
obs, reward, terminated, truncated, info = env.step(0)
for step in range(6000):
    # Mario's position
    x, _ = mario_loc(obs)
    y = info["y_pos"]
    # print(f"Mario ({x} | {y})")

    x_enemy, y_enemy = exist_enemy(obs)
    if x_enemy is not None:
        # print(f"enemy ({x_enemy} | {y_enemy})")
        if (x_enemy - x < 32 and x_enemy - x > 28) and y <= 79:
            print(f"Mario ({x} | {y})")
            print(f"enemy ({x_enemy} | {y_enemy})")
            obs, reward, terminated, truncated, info = env.step(2)
            time.sleep(delay)
            continue
    

    # x_pipe, y_pipe = exist_pipe(obs)

    pipe_values = exist_pipe(obs)
    print(x)
    x_pipe, y_pipe = find_nearest_pipe(x,y,pipe_values)


    if x_pipe is not None:
        # print(f"pipe ({x_pipe} | {y_pipe})")
        if y_pipe == 184: # short pipe
            print("short pipe")
            if (x_pipe - x < 45 and x_pipe - x > 40) and y <= 79:
                print(f"Mario ({x} | {y})")
                print(f"pipe ({x_pipe} | {y_pipe})")
                env.step(5)
                time.sleep(delay)
                obs, reward, terminated, truncated, info = env.step(1)
                time.sleep(delay)
                continue
        elif y_pipe == 168: # medium pipe
            print("medium pipe")
            print(f"pipe ({x_pipe} | {y_pipe})")
            print(f"Mario ({x} | {y})")
            if (x_pipe - x < 65 and x_pipe - x > 27) and y <= 79:
                print(f"Mario ({x} | {y})")
                print(f"pipe ({x_pipe} | {y_pipe})")
                for _ in range(50):
                    env.step(2)
                    time.sleep(delay)
                obs, reward, terminated, truncated, info = env.step(1)
                time.sleep(delay)
                continue
        else: # long pipe
            print("long pipe")
            print(f"pipe ({x_pipe} | {y_pipe})")
            if (x_pipe - x < 75 and x_pipe - x > 27) and y <= 79:
                print(f"Mario ({x} | {y})")
                print(f"pipe ({x_pipe} | {y_pipe})")
                for _ in range(20):
                    obs, reward, terminated, truncated, info = env.step(5)
                    time.sleep(delay)
                obs, reward, terminated, truncated, info = env.step(1)
                time.sleep(delay)
                continue

    obs, reward, terminated, truncated, info = env.step(1)
    time.sleep(delay)


    done = terminated or truncated
    
env.close()


# obs_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

# cv2.imshow("obs", obs_img)
# cv2.waitKey(0)

# low = np.array([0, 168, 0])
# high = np.array([1, 168, 10])
# mask = cv2.inRange(obs_img, low, high)

# cv2.imshow("mask", mask)
# cv2.waitKey(0)

# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cont_img = cv2.drawContours(obs_img, contours, -1, (0, 0, 255), 2)

# # Area of pipe: 137.5
# for contour in contours:
#     print(cv2.contourArea(contour))
#     if cv2.contourArea(contour) < 140 and cv2.contourArea(contour) > 135:
#         x = contour[0][0][0]
#         y = contour[0][0][1]
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(obs_img, (x, y), (x + w, y + h), (255, 0, 0), 2)


# cv2.imshow("obs_img", obs_img)
# cv2.waitKey(0)

# cv2.destroyAllWindows()