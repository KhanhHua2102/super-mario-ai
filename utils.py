import hashlib
import numpy as np
import cv2


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
