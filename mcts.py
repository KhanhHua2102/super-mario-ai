import math
import warnings

import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from mario_states import Mario_States


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
        node = max(node.children, key=lambda n: (n.value / (n.visits + 1)) + math.sqrt(math.log(node.visits + 1) / (n.visits + 1)))
    return node


def expand(node: Node) -> Node:
    """
    This function is called when a node is selected for expansion.
    It randomly chooses an untried action from the current node's state 
    and creates a new node with the resulting state. 
    This new node is added as a child of the current node.
    """
    # actions = node.state.get_actions()
    action = env.action_space.sample()
    new_state = Mario_States(*env.step(action))
    new_node = Node(new_state, parent=node)
    node.children.append(new_node)
    return new_node


def simulate(node: Node) -> int:
    """
    The simulation function is used to play out a game from the current state in a completely random manner.
    It continues to select random actions until a terminal state is reached,
    and then returns the result of that terminal state.
    """
    state = node.state.copy_state()
    while not state.is_terminal():
        action = env.action_space.sample()
        state = Mario_States(*env.step(action))
    return state.get_reward()


def backpropagate(node: Node, value: int) -> None:
    """
    After a simulation is performed, the backpropagation function updates 
    the statistics of nodes in the path from the expanded node to the root.
    It increments the visit count and updates the value of each node based on the simulation result.
    This process helps to accumulate information about the quality of different actions.
    """
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
        return max(node.children, key=lambda n: (n.value / n.visits) + math.sqrt(2 * math.log(node.visits) / n.visits))
    else:
        return node


def MCTS(root_node: Node, iterations: int) -> Mario_States:
    """
    This is the main function that implements the MCTS algorithm. 
    It takes the root state of the game and the number of iterations as inputs and 
    returns the best state found by MCTS after the specified number of iterations.
    """
    for iteration in range(iterations):
        print(iteration)

        env.reset()

        node = select(root_node)
        if not node.state.is_terminal():
            if len(node.children) < env.action_space.n:
                node = expand(node)
            else:
                node = select(node)
        
        reward = simulate(node)
        backpropagate(node, reward)

        print(reward)

    results = []
    # construct list of children from root node to leaf node
    while root_node.children:
        root_node = best_child(root_node)
        results.append(root_node.state)
    
    # return the state of the leaf node with the highest value
    return max(results, key=lambda n: n.reward)


warnings.filterwarnings("ignore")

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.reset()
obs, reward, terminated, truncated, info = env.step(0)


iterations = 2

root = Node(state=Mario_States(obs, reward, terminated, truncated, info))
result = MCTS(root, iterations)

print(result)