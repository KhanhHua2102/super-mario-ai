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


warnings.filterwarnings("ignore")  # Suppress all warnings
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = gym.make(
    "SuperMarioBros-1-1-v3", apply_api_compatibility=True, render_mode="rgb_array"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

env.reset()

# ----------------------------------------------------------

TOTAL_TIMESTEPS = 1000000
CHECK_FREQ = 10000
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


# ----------------------------------------------------------

CHECKPOINT_DIR = "./PPO/train/"
LOG_DIR = "./PPO/logs/"

callback = TrainAndLoggingCallback(check_freq=CHECK_FREQ, save_path=CHECKPOINT_DIR)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
)

# ----------------------------------------------------------

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

model.save("thisisatestmodel")
