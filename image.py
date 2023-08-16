#show image for the specified observations

import numpy as np
import scipy.misc as smp
from PIL import Image
import cv2

def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return grayscale_image

def normalize_grayscale(image):
    # Normalize to [0, 1]
    normalized_image = image / 255.0 
    return normalized_image

#visualing the state at a particular step as an greyscale image
def produce_image(obs):
    data = np.zeros( (240,256,3), dtype=np.uint8 )
    for i in range(len(obs)):
        for j in range(len(obs[i])):
            data[i][j] = obs[i][j]

    # image = Image.fromarray(data)
    # grey_scale_image = convert_to_grayscale(image)
    # grey_scale_image.show()

    grey_scale_image = convert_to_grayscale(data)
    image = Image.fromarray(grey_scale_image)
    image.show()
#     # View in default viewer
    # image.show()  

