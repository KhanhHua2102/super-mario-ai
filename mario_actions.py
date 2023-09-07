import time

CUSTOM_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["A"],
    ["right", "B"],
]


def low_jump(env, times, delay):
    obs, reward, terminated, truncated, info = None, None, None, None, None
    for _ in range(times):
        env.step(2)
        time.sleep(delay)
        # wait for mario to fall
        for _ in range(14):
            env.step(0)
            time.sleep(delay)
        obs, reward, terminated, truncated, info = env.step(0)

    return obs, reward, terminated, truncated, info


def high_jump(env, times, delay):
    obs, reward, terminated, truncated, info = None, None, None, None, None
    for _ in range(times):
        for _ in range(20):
            env.step(2)
            time.sleep(delay)
        # wait for mario to fall
        for _ in range(17):
            env.step(0)
            time.sleep(delay)
        obs, reward, terminated, truncated, info = env.step(0)

    return obs, reward, terminated, truncated, info
