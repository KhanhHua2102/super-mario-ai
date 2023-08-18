class Mario_States:
    def __init__(self, obs, reward, terminated, truncated, info):
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info

    def __str__(self):
        return "\nreward: {}\nterminated: {}\ntruncated: {}\ninfo: {}".format(
            self.reward, self.terminated, self.truncated, self.info)
    
    def is_terminal(self) -> bool:
        return self.terminated or self.truncated
    
    def get_reward(self) -> int:
        return self.reward

    def copy_state(self) -> "Mario_States":
        new_state = Mario_States(self.obs, self.reward, self.terminated, self.truncated, self.info)
        return new_state
    
