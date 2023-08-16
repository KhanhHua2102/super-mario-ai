from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT
import gym
import image

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = JoypadSpace(env,COMPLEX_MOVEMENT)

# Makes the middle pixel red
# data[512,512] = [254,0,0]       
# Makes the next pixel blue
# data[512,513] = [0,0,255]       


done = True
env.reset()
for step in range(10):
    action = env.action_space.sample()
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    # 240 256 3 dimension of the pixels values
    if step == 1:
        image.produce_image(obs)


    done = terminated or truncated

    if done:
        state = env.reset()

env.close()

print("hello world")