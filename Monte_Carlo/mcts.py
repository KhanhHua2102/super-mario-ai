import math
import warnings
import time
from typing import Union
import random

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from Monte_Carlo.mario_states import Mario_States, Node


CUSTOM_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["A"],
]

ACTION_SIZE = len(CUSTOM_MOVEMENT)


warnings.filterwarnings("ignore")  # Suppress all warnings


def select(node: Node) -> Node:
    """
    This function is responsible for navigating down the tree to select a node for expansion.
    It follows the UCB1 formula (Upper Confidence Bound) to balance exploration and exploitation.
    The idea is to select nodes that are either promising
    (high value and low visits) or unexplored (not fully expanded).
    This function iteratively moves down the tree
    until it reaches a leaf node that is either unexplored or terminal.
    """
    while node.children:
        node = max(
            node.children,
            key=lambda n: (n.value / n.visits if n.visits > 0 else 0)
            + math.sqrt(2 * math.log(node.visits) / (n.visits)),
        )
        env.step(node.state.action)
        time.sleep(delay)
    return node


def expand(node: Node) -> Union[Node, None]:
    """
    This function is called when a node is selected for expansion.
    It randomly chooses an untried action from the current node's state
    and creates a new node with the resulting state.
    This new node is added as a child of the current node.
    """
    action = random.choice(node.untried_actions())
    new_state = Mario_States(*env.step(action), action)
    time.sleep(delay)
    new_node = Node(new_state)
    new_node.add_parent(node)
    node.add_child(new_node)
    return new_node


def simulate(node: Node, action_limit: int, stuck_limit: int) -> int:
    """
    The simulation function is used to play out a game from the current state in a completely random manner.
    It continues to select random actions until a terminal state is reached,
    and then returns the result of that terminal state.
    """
    reward = 0
    noreward_count = 0

    state = node.state.copy_state()
    while not state.is_terminal() and action_limit > 0:
        action = env.action_space.sample()
        state = Mario_States(*env.step(action), action)
        time.sleep(delay)
        reward += state.get_reward()
        action_limit -= 1
        if state.get_reward() == 0:
            noreward_count += 1

        if state.get_reward() > 0:
            noreward_count = 0

        if noreward_count > stuck_limit:
            action_limit = 0
    return reward


def backpropagate(node: Node, value: int) -> None:
    """
    After a simulation is performed, the backpropagation function updates
    the statistics of nodes in the path from the expanded node to the root.
    It increments the visit count and updates the value of each node based on the simulation result.
    This process helps to accumulate information about the quality of different actions.
    """
    if node.parent is None:
        return
    while node:
        node.visits += 1
        node.value += value
        node = node.parent

    return None


def best_child(node: Node) -> Node:
    """
    This function selects the child node with the highest value according to the UCB1 formula.
    It balances exploration and exploitation by considering
    both the node's estimated value and the number of visits.
    """
    if node.children:
        return max(
            node.children,
            key=lambda n: (n.value / n.visits if n.visits > 0 else 0)
            + math.sqrt(2 * math.log(node.visits) / (n.visits)),
        )
    else:
        return node


def MCTS(
    root_node: Node,
    iterations: int,
    limit_per_iteration: int,
    limit_per_simulation: int,
    stuck_limit: int,
):
    """
    This is the main function that implements the MCTS algorithm.
    It takes the root state of the game and the number of iterations as inputs and
    returns the best state found by MCTS after the specified number of iterations.
    """
    for iteration in range(iterations):
        print("\niteration:", iteration + 1)

        node = root_node

        env.reset()

        # selection
        while not node.state.is_terminal() and len(node.children) == ACTION_SIZE:
            node = select(node)

        # expansion
        if not node.state.is_terminal():
            node = expand(node)

        # simulation
        reward = simulate(node, limit_per_simulation, stuck_limit)

        print("reward:", reward)

        # backpropagation
        backpropagate(node, reward)

    # Select the best child from the root node
    results = []
    node = root_node
    while node.children:
        node = best_child(node)
        print(node)
        results.append((node.state.action, node.value, node.visits))

    return results


env = gym.make(
    "SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="rgb_array"
)
env = JoypadSpace(env, CUSTOM_MOVEMENT)
env.reset()

obs, reward, terminated, truncated, info = env.step(0)


delay = 0
iterations = 10000
action_limit_per_iteration = 100
action_limit_per_simulation = 1500
stuck_limit = 60

root = Node(state=Mario_States(obs, reward, terminated, truncated, info, 0))
results = MCTS(
    root,
    iterations,
    action_limit_per_iteration,
    action_limit_per_simulation,
    stuck_limit,
)

action_list = []
print("\nRESULT:\n")
for state in results:
    print(state)
    action_list.append(state[0])

# export action list to file
with open("Monte_Carlo/action_list.txt", "w") as f:
    for action in action_list:
        f.write(str(action) + "\n")
