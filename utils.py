import json
import numpy as np


class Stat(object):
    """
    This class is used to store the statistics of the training and running process.
    Including the number of actions, total reward and heatmap.
    """

    def __init__(self):
        self.nums_action = 0
        self.total_reward = [0]
        self.heatmap = np.zeros((256, 240))

    def update(self, total_reward, mario_x, mario_y):
        self.nums_action += 1
        self.total_reward.append(
            self.total_reward[len(self.total_reward) - 1] + total_reward
        )
        self.heatmap[mario_y, mario_x] += 1

    def get_nums_action(self):
        return self.nums_action

    def get_total_reward(self):
        return self.total_reward

    def get_heatmap(self):
        return self.heatmap


def print_stats(
    curr_episode, curr_total_rewards, curr_time, explore_rate, learning_rate
):
    """
    This function is used to print the statistics of the training process.
    """
    print("-------------------------")
    print("Episode: " + str(curr_episode + 1))
    print("Score: " + str(curr_total_rewards))
    print("Time: " + str(curr_time))
    print("Explore Rate: " + str(explore_rate))
    print("Learning Rate: " + str(learning_rate))
    print("-------------------------")


def save_q_table(q_table, episode, learning_rate, exploration_rate, file_name):
    """
    This function is used to save the q_table and the statistics of the training process
    inorder to continue the training process.
    """
    with open(
        f"Q_Learning/train/current_model_{file_name}.json", "w+", encoding="utf-8"
    ) as json_file:
        json.dump(str(q_table), json_file)
    with open(
        f"Q_Learning/train/model_statistics_{file_name}.json", "w+", encoding="utf-8"
    ) as file:
        data = {
            "iterations": episode,
            "learning_rate": learning_rate,
            "exploration_rate": exploration_rate,
        }
        json.dump(data, file)
