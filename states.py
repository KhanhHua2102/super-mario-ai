import cv2
import numpy as np

import contours_detector as cd


# 1: small pipe
# 2: medium pipe
# 3: large pipe
# 4: near small pipe
# 5: near medium pipe
# 6: near large pipe
# 7: 1 enemy
# 8: 2 enemies
# 9: 1 and 2 enemies
# 10: 2 and 1 enemies
# 11: 3 enemies
# 12: hole


def get_state(obs):
    small_pipe, medium_pipe, large_pipe, near_small_pipe, near_medium_pipe, near_large_pipe, one_enemy, two_enemies, one_and_two_enemies, two_and_one_enemies, three_enemies, hole = False, False, False, False, False, False, False, False, False, False, False, False
    
    if small_pipe:
        return 0
    elif medium_pipe:
        return 1
    elif large_pipe:
        return 2
    elif near_small_pipe:
        return 3
    elif near_medium_pipe:
        return 4
    elif near_large_pipe:
        return 5
    elif one_enemy:
        return 6
    elif two_enemies:
        return 7
    elif one_and_two_enemies:
        return 8
    elif two_and_one_enemies:
        return 9
    elif three_enemies:
        return 10
    elif hole:
        return 11
    else:
        return 12
