import detectors as cd

# state space:
# 0: small pipe in range
# 1: medium pipe in range
# 2: large pipe in range
# 3: 1 enemy in range
# 4: hole in range
# 5: nothing in range


# Return Mario's relative and absolute position
def mario_location(obs, info) -> tuple[int, int]:
    x_mario, _ = cd.mario_loc_detect(obs)
    y_mario = info["y_pos"]
    return x_mario, y_mario


def one_enemy_detect(obs, info) -> bool:
    x_mario, y_mario = mario_location(obs, info)

    # Goomba
    enemy_locs = cd.exist_enemy(obs)
    if len(enemy_locs) > 0:
        loc_list = [loc for loc in enemy_locs if loc[0] > x_mario]
        if len(loc_list) > 0:
            loc = min(loc_list)
            x_enemy = loc[0]
            y_enemy = loc[1]
        else:
            x_enemy, y_enemy = None, None
    else:
        x_enemy, y_enemy = None, None

    if (
        x_enemy is not None
        and (x_enemy - x_mario < 32 and x_enemy - x_mario > 28)
        and y_mario == 79
        and y_enemy == 198
    ):
        return True

    # Turtle
    turle_x, turle_y = cd.exist_turtle(obs)
    if (
        turle_x is not None
        and (turle_x - x_mario < 33 and turle_x - x_mario > 28)
        and y_mario == 79
        and turle_y == 195
    ):
        return True
    return False


def hole_detect(obs, info) -> bool:
    x_mario, y_mario = mario_location(obs, info)
    small_hole = cd.exist_small_hole(obs)
    if (
        small_hole != (None, None)
        and small_hole[0] - x_mario <= 3
        and small_hole[0] - x_mario > 0
        and y_mario == 79
    ):
        return True
    return False


def small_pipe_detect(obs, info) -> bool:
    x_mario, y_mario = mario_location(obs, info)
    pipe_values = cd.exist_pipe(obs)
    x_pipe, y_pipe = cd.find_nearest_pipe(x_mario, pipe_values)
    if x_pipe is not None:
        if y_pipe == 184:  # short pipe
            if (x_pipe - x_mario < 45 and x_pipe - x_mario > 40) and y_mario == 79:
                return True
    return False


def medium_pipe_detect(obs, info) -> bool:
    x_mario, y_mario = mario_location(obs, info)
    pipe_values = cd.exist_pipe(obs)
    x_pipe, y_pipe = cd.find_nearest_pipe(x_mario, pipe_values)
    if x_pipe is not None:
        if y_pipe == 168:  # medium pipe
            if (x_pipe - x_mario < 65 and x_pipe - x_mario > 27) and y_mario == 79:
                return True
    return False


def large_pipe_detect(obs, info) -> bool:
    x_mario, y_mario = mario_location(obs, info)
    pipe_values = cd.exist_pipe(obs)
    x_pipe, y_pipe = cd.find_nearest_pipe(x_mario, pipe_values)
    if x_pipe is not None:
        if y_pipe != 184 and y_pipe != 168:  # large pipe
            if (x_pipe - x_mario < 75 and x_pipe - x_mario > 27) and y_mario == 79:
                return True
    return False


def get_state(obs, info) -> int:
    small_pipe, medium_pipe, large_pipe, one_enemy, hole = (
        small_pipe_detect(obs, info),
        medium_pipe_detect(obs, info),
        large_pipe_detect(obs, info),
        one_enemy_detect(obs, info),
        hole_detect(obs, info),
    )

    if small_pipe:
        return 0
    elif medium_pipe:
        return 1
    elif large_pipe:
        return 2
    elif one_enemy:
        return 3
    elif hole:
        return 4
    else:
        return 5
