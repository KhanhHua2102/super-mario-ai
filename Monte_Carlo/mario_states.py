CUSTOM_MOVEMENT = [["NOOP"], ["right"], ["right", "A"], ["A"]]


class Mario_States:
    """
    Class representing the state of one node in the tree.
    """

    def __init__(self, obs, reward, terminated, truncated, info, action):
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.action = action

    def __str__(self):
        return "reward: {}\nterminated: {}\ntruncated: {}\ninfo: {}".format(
            self.reward, self.terminated, self.truncated, self.info
        )

    def is_terminal(self) -> bool:
        return self.terminated or self.truncated

    def get_reward(self) -> int:
        return self.reward

    def copy_state(self) -> "Mario_States":
        new_state = Mario_States(
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.action,
        )
        return new_state

    def untried_actions(self):
        return CUSTOM_MOVEMENT - list(self.children.action.keys())


class Node:
    """
    This class represents a node in the tree.
    Each node contains the state of the game,
    a reference to its parent node, a list of its children,
    and variables to keep track of the number of visits and the accumulated value.
    """

    def __init__(self, state: Mario_States, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def add_child(self, node):
        self.children.append(node)

    def add_parent(self, node):
        self.parent = node

    def untried_actions(self):
        action_list = []
        for node in self.children:
            action_list.append(node.state.action)
        return list(set([0, 1, 2, 3]) - set(action_list))

    def __str__(self):
        return "\nno child: {}\nvisits: {}\nvalue: {}\n".format(
            len(self.children), self.visits, self.value
        )
