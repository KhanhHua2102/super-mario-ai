import warnings

from stable_baselines3 import PPO


warnings.filterwarnings("ignore")
model = PPO.load("./PPO/train/best_model_1000000.zip")
