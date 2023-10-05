# Import the game
import json
import os
import warnings

import gym
import gym_super_mario_bros
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

warnings.filterwarnings("ignore")
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# ----------------------------------------------------------

# 1. Create the base environment
env = gym.make(
    "SuperMarioBros-1-1-v3", apply_api_compatibility=True, render_mode="rgb_array"
)
# 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order="last")

env.reset()

# ----------------------------------------------------------

CHECKPOINT_DIR = "./PPO/train/"
LOG_DIR = "./PPO/logs/"

CHECK_FREQ = 10000
TOTAL_TIMESTEPS = 100000

TIME_STEPS = 10000

LEARNING_RATE = 0.0001
N_STEPS = 512  # Steps before update network model

# ----------------------------------------------------------


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, "best_model_{}".format(self.n_calls)
            )
            self.model.save(model_path)

        return True


# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=CHECK_FREQ, save_path=CHECKPOINT_DIR)

current_iteration = 0
conf = None
if os.path.exists("PPO/train/model_statistics"):
    with open("PPO/train/model_statistics", "r") as f:
        conf = json.load(f)
        current_iteration = conf["iterations"]

        print(f"\nLoaded model from iteration {current_iteration}\n")
else:
    print("\nNo model found, starting from scratch\n")

# ----------------------------------------------------------

model_load_file = "PPO/train/current_best_model"
if os.path.exists(model_load_file):
    # This is the AI model started
    model = PPO.load(model_load_file, env=env)
else:
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
    )

# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

model.save(f"PPO/train/current_best_model")
model.save(f"PPO/train/best_model_{current_iteration+TIME_STEPS}")

with open("PPO/train/model_statistics", "w") as f:
    conf["iterations"] += TIME_STEPS
    json.dump(conf, f)
