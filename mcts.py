import math
import warnings
import copy
import time
from typing import Union
import random

import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack
from nes_py.wrappers import JoypadSpace

from mario_states import Mario_States, CUSTOM_MOVEMENT


warnings.filterwarnings("ignore")



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

    def add_parent(self,node):
        self.parent = node

    def __str__(self):
        return "no child: {}\nvisits: {}\nvalue: {}".format(
            len(self.children), self.visits, self.value)


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

    action = env.action_space.sample()
    # action = random.choice(CUSTOM_MOVEMENT)
    print("untried action:",action)
    if action not in node.children:
        new_state = Mario_States(*env.step(action), action)
        time.sleep(delay)
        new_node = Node(new_state)
        new_node.add_parent(node)
        return new_node
    return None
    


def simulate(node: Node, action_limit: int) -> int:
    """
    The simulation function is used to play out a game from the current state in a completely random manner.
    It continues to select random actions until a terminal state is reached,
    and then returns the result of that terminal state.
    """
    print("simulating")
    reward = 0
    # state = node.state.copy_state()
    state = copy.deepcopy(node.state)
    while not state.is_terminal() and action_limit > 0:
        if action_limit % 100 == 0:
            print("action limit:", action_limit)
        action = env.action_space.sample()
        state = Mario_States(*env.step(action), action)
        time.sleep(delay)
        reward += state.get_reward()
        action_limit -= 1
    print("end of simulation\n")
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
        return max(node.children, key=lambda n: (n.value / n.visits) + math.sqrt(2 * math.log(node.visits) / n.visits))
    else:
        return node


def MCTS(root_node: Node, iterations: int, limit_per_iteration: int, limit_per_simulation: int):
    """
    This is the main function that implements the MCTS algorithm. 
    It takes the root state of the game and the number of iterations as inputs and 
    returns the best state found by MCTS after the specified number of iterations.
    """
    root = root_node
    for iteration in range(iterations):
        print("\niteration:", iteration + 1)
        limit = limit_per_iteration

        # env.reset()

        node = select(root)
        print("rootnode child:", len(root.children))
        
        print("selecting process:")
        
        # ISSUE: NEED TO GO DOWN THE TREE UNTIL REACH THE LEAF NOD
        while not node.state.is_terminal() and limit > 0:
            print("limit:", limit)
            if len(node.children) < env.action_space.n:
                print("expanding")
                # time.sleep(10)
                print("rootnode child:", len(root.children))
                new_node = expand(node)
                if new_node is not None:
                    node.add_child(new_node) 
                    time.sleep(2)
                # ISSUE: ROOT_NODE CHILDREN DOES NOT UPDATED !!!
                # can be updated
                print("rootnode child:", len(root.children))
            else:
                print("selecting")
                node = select(node)
            limit -= 1
        
       


        print("end of selection\n")

        reward = simulate(node, limit_per_simulation)

        print("reward:",reward)

        backpropagate(node, reward)
        print("rootnode child:", len(root.children))

        print(reward)

    results = []
    # construct list of children from root node to leaf node
    # ISSUE: NOT RETURN THE CORRECT OPTIMAL PATH
    while root.children:
        root = best_child(root)
        results.append((root.visits, root.value, root.state.action))


    
    # return the state of the leaf node with the highest value
    return results
    # return best_child(root).state.action




env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, CUSTOM_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)

env.reset()
obs, reward, terminated, truncated, info = env.step(0)


delay = 0
iterations = 100
action_limit_per_iteration = 1500
action_limit_per_simulation = 150

root = Node(state=Mario_States(obs, reward, terminated, truncated, info, 0))
results = MCTS(root, iterations, action_limit_per_iteration, action_limit_per_simulation)

new_state, reward, done,_,_ = env.step(results)
print("this is result action:",results)


action_list = []
print("\nRESULT:\n")
for state in results:
    print(state[:2])
    action_list.append(state[2])

# export action list to file
with open("action_list.txt", "w") as f:
    for action in action_list:
        f.write(str(action) + "\n")
