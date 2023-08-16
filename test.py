from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

from gym.wrappers.gray_scale_observation import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Create the base environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
# Wrap env in a DummyVecEnv
env = DummyVecEnv([lambda: env])
# Stack the frames
env = VecFrameStack(env, n_stack=4)

done = True

env.reset()

for _idx in range(1000):
    obs, reward, done, info = env.step([1])

    if done:
        state = env.reset()
    
env.close()