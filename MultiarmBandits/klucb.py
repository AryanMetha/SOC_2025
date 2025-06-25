import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt


def kl_divergence(p, q):
    eps = 1e-15  # to avoid log(0)
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

# Binary search to solve: n * KL(p || q) <= bound
def solve_kl_ucb(p, n, bound, tol=1e-4):
    low = p
    high = 1.0
    while high - low > tol:
        mid = (low + high) / 2
        if n * kl_divergence(p, mid) > boun d:
            high = mid
        else:
            low = mid
    return low  # the largest q satisfying the inequality



class KLUCBAgent(Agent):
    # Add fields 
    reward_memory: np.ndarray  # A per arm value of how much reward was gathered
    count_memory: np.ndarray 
    upper_bounds: np.ndarray  # Upper bounds for each arm based on KL divergence



    def __init__(self, time_horizon, bandit:MultiArmedBandit,): 
        # Add fields
        super().__init__(time_horizon, bandit)
        self.bandit: MultiArmedBandit = bandit
        self.reward_memory = np.zeros(len(bandit.arms))
        self.count_memory = np.zeros(len(bandit.arms))
        self.upper_bounds = np.zeros(len(bandit.arms))
        self.time_step = 0


    def give_pull(self):
        # Choose the arm with the highest upper bound
        chosen_arm = np.argmax(self.upper_bounds)
        reward = self.bandit.pull(chosen_arm)
        self.reinforce(reward, chosen_arm)  # Reinforce the agent with the received reward
        

    def reinforce(self, reward, arm):
        self.count_memory[arm] += 1
        self.reward_memory[arm] += reward
        self.time_step += 1
        self.rewards.append(reward)

        # Update the upper bounds for each arm based on KL divergence
        for arm in range(len(self.bandit.arms)):
            if self.count_memory[arm] > 0:
                self.upper_bounds[arm] = solve_kl_ucb(self.bandit.arms[arm], self.count_memory[arm], np.log(self.time_step))
            else:
                self.upper_bounds[arm] = float('inf') #force initial exploration

 
    def plot_arm_graph(self):
        counts = self.count_memory
        indices = np.arange(len(counts))

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.bar(indices, counts, color='skyblue', edgecolor='black')

        # Formatting
        plt.title('Counts per Category', fontsize=16)
        plt.xlabel('Arm', fontsize=14)
        plt.ylabel('Pull Count', fontsize=14)
        plt.grid(axis='y', linestyle='-')  # Add grid lines for the y-axis
        plt.xticks(indices, [f'Category {i+1}' for i in indices], rotation=45, ha='right')
        # plt.yticks(np.arange(0, max(counts) + 2, step=2))

        # Annotate the bars with the count values
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12, color='black')

        # Tight layout to ensure there's no clipping of labels
        plt.tight_layout()

        # Show plot
        plt.show()

# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent = KLUCBAgent(TIME_HORIZON,bandit) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()
        if agent.time_step % 1000 == 0:
            print(f"Step {agent.time_step}: Rewards = {agent.rewards[-10:]}")

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
