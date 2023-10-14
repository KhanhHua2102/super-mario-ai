import os
import time
import warnings
import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from matplotlib import pyplot as plt
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from utils import Stat
from Image_Detection.detectors import mario_loc_detect

warnings.filterwarnings("ignore")
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
# 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order="last")


model = PPO.load("PPO/train/Current Best Model.zip")
state = env.reset()

done = False
death = 0
start = time.time()
stats = Stat()

while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)

    mario_x, mario_y = mario_loc_detect(state[0])
    stats.update(reward, mario_x, mario_y)

    if done:
        death += 1
        print("Death:", death)

    if info[0]["stage"] == 2:
        print("\nFinished stage 1\n")
        break

    env.render()


# Agent Stats
end = time.time()
print("Time of execution:", round((end - start) * 10**3, 1), "ms")
print("Time left in game: ", info[0]["time"], "ms")
print("Score: ", info[0]["score"])
print("Number of actions: ", stats.get_nums_action())
print("FPS: ", round(stats.get_nums_action() / (end - start), 1))
print("Furthest distance: ", info[0]["x_pos"])
print("Total reward: ", stats.get_total_reward()[-1])
print()

# ------------------------------------------------------------

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
# plt.gca().invert_yaxis()
plt.title("Agent's Activity Heatmap")
plt.show()
