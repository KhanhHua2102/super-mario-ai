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