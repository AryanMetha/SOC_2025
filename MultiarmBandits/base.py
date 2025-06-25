import numpy as np
import random
import matplotlib.pyplot as plt

class MultiArmedBandit:
    def __init__(self, arms : np.ndarray[float]):
        self.arms = arms # list of probabilities of each arm returning a favorable reward
        self.best_arm = np.max(arms) # useful for regret calculation
        self.cumulative_regret_array = [0]
        self.reward_array=[0]

    def pull(self, arm:int) -> int:
        assert arm in np.arange(0, len(self.arms)), "Action undefined for bandit" # check if arm index is valid
        reward = 1 if np.random.random() < self.arms[arm] else 0
        self.cumulative_regret_array.append(self.cumulative_regret_array[-1] + self.best_arm - reward)
        self.reward_array.append(reward)
        return reward
    
    def plot_cumulative_regret(self):
        timesteps = np.arange(1, len(self.cumulative_regret_array) + 1)

        # Plot the data
        plt.figure(figsize=(8,4))
        plt.plot(timesteps, self.cumulative_regret_array, linestyle='-', color='r', label='Cumulative Regret')

        # Formatting
        plt.title('Cumulative Regret Over Time', fontsize=16)
        plt.xlabel('Timesteps', fontsize=14)
        plt.ylabel('Cumulative Regret', fontsize=14)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5)
        plt.yticks(np.arange(0, max(self.cumulative_regret_array) + 5, step=5))

        # Add legend
        plt.legend(loc='upper left', fontsize=12)

        # Tight layout to ensure there's no clipping of labels
        plt.tight_layout()

        # Show plot
        plt.show()



class Agent:
    def __init__(self, time_to_run, bandit : MultiArmedBandit):
        self.time_to_run = time_to_run
        self.rewards = []
        self.bandit = bandit
        self.arms = len(bandit.arms)
    
    def plot_reward_vs_time_curve(self):

        # Create an index for timesteps
        timesteps = np.arange(1, len(self.rewards) + 1)

        # Average out self.rewards
        avg_rewards = [np.mean(self.rewards[0:T+1]) for T in range(self.time_to_run)]

        # Plot the data
        plt.figure(figsize=(8,4))
        plt.plot(timesteps, avg_rewards, linestyle='-', color='g', label='Rewards')

        # Formatting
        plt.title('Average Reward Over Time', fontsize=16)
        plt.xlabel('Timesteps', fontsize=14)
        plt.ylabel('Mean Reward Value upto timestep t', fontsize=14)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5)
        # plt.xticks(timesteps)  # Show all timesteps as x-axis ticks
        # plt.yticks(np.arange(0, max(self.rewards) + 5, step=5))

        # Add legend
        plt.legend(loc='upper left', fontsize=12)

        # Tight layout to ensure there's no clipping of labels
        plt.tight_layout()

        # Show plot
        plt.show()
