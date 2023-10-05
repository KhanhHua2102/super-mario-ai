import hashlib
import json

import cv2
import numpy as np


def stack_frames(frames, new_frame, stack_size):
    if frames is None:
        frames = np.zeros((stack_size, *new_frame.shape), dtype=np.uint8)

    frames[:-1] = frames[1:]
    frames[-1] = new_frame

    return frames


def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return grayscale_image


def hashState(obs, action) -> int:
    # Convert the 3D array to a string representation
    obs_str = str(obs)
    action_str = str(action)
    arr_str = obs_str + action_str

    # Use a hash function to generate a unique hash for the array
    hash_obj = hashlib.md5(arr_str.encode())

    # Convert the hash to an integer (you can choose a different method if needed)
    hash_int = int(hash_obj.hexdigest(), 16)

    return hash_int


def print_stats(
    curr_episode, curr_total_rewards, curr_time, explore_rate, learning_rate
):
    print("-------------------------")
    print("Episode: " + str(curr_episode + 1))
    print("Score: " + str(curr_total_rewards))
    print("Time: " + str(curr_time))
    print("Explore Rate: " + str(explore_rate))
    print("Learning Rate: " + str(learning_rate))
    print("-------------------------")


def save_q_table(q_table, episode, file_name):
    # Save q_table to json file
    with open(
        f"Q_Learning/train/current_model{file_name}.json", "w+", encoding="utf-8"
    ) as json_file:
        json.dump(q_table, json_file)
    with open(
        f"Q_Learning/train/model_statistics{file_name}.json", "w+", encoding="utf-8"
    ) as file:
        json.dump({"iterations": episode + 1}, file)
