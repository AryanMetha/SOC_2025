import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt


class ThompsonAgent(Agent):
    # Add fields 
    reward_memory: np.ndarray  # A per arm value of how much reward was gathered
    count_memory: np.ndarray  # An array of the number of times an arm is pulled
    p_distributions: list  # List of beta distributions for each arm
    prior_params: list  # List of prior parameters for each arm (alpha, beta)


    def __init__(self, time_horizon, bandit:MultiArmedBandit,): 
        # Add fields
        super().__init__(time_horizon, bandit)
        self.bandit: MultiArmedBandit = bandit
        self.reward_memory = np.zeros(len(bandit.arms))
        self.count_memory = np.zeros(len(bandit.arms))
        self.prior_params = [(1, 1) for _ in range(len(bandit.arms))]
       # self.p_distributions = [np.random.beta(a, b) for a, b in self.prior_params]
        self.time_step = 0


    def give_pull(self):
        # Sample from the beta distributions for each arm
        sampled_values = [np.random.beta(*self.prior_params[arm]) for arm in range(len(self.bandit.arms))]
        
        # Choose the arm with the highest sampled value
        chosen_arm = np.argmax(sampled_values)
        
        # Pull the chosen arm and get the reward
        reward = self.bandit.pull(chosen_arm)
        
        # Reinforce the agent with the received reward
        self.reinforce(reward, chosen_arm)
         
        # prob_reward = self.bandit.arms[chosen_arm]
        # reward = 1 if np.random.random() < prob_reward else 0
        # self.reinforce(reward, chosen_arm)


    def reinforce(self, reward, arm):
        self.count_memory[arm] += 1
        self.reward_memory[arm] += reward
        self.time_step += 1
        self.rewards.append(reward)

        # Update the prior parameters for the chosen arm
        alpha, beta = self.prior_params[arm]
        self.prior_params[arm] = (alpha + reward, beta + (1 - reward))
        #self.p_distributions[arm] = np.random.beta(*self.prior_params[arm])
 
    def plot_arm_graph(self):
        raise NotImplementedError


# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent = ThompsonAgent(TIME_HORIZON,bandit) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()

