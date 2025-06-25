import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt


class UCB(Agent):
    # Add fields 
    confidence: float  # Confidence level for the Thompson sampling
    reward_memory: np.ndarray  # A per arm value of how much reward was gathered
    count_memory: np.ndarray  # An array of the number of times an arm is pulled
    buffed_reward_memory : np.ndarray # An array of the average reward per arm+ the bonus factor 

    def __init__(self, time_horizon, bandit:MultiArmedBandit,confidence=1.0): 

        # Add fields
        super().__init__(time_horizon, bandit)
        self.confidence = confidence
        self.bandit: MultiArmedBandit = bandit
        self.reward_memory = np.zeros(len(bandit.arms))
        self.count_memory = np.zeros(len(bandit.arms))
        self.buffed_reward_memory = np.zeros(len(bandit.arms))
        self.time_step = 0




    def give_pull(self):
        chosen_arm = np.argmax(self.buffed_reward_memory)  # Choose the arm with the highest buffed reward
        reward = self.bandit.pull(chosen_arm)  # Pull the chosen arm
        self.reinforce(reward, chosen_arm)  # Reinforce the agent with the received reward
        # Update the buffed reward memory with the bonus factor


    def reinforce(self, reward, arm):
        self.count_memory[arm] += 1
        self.reward_memory[arm] += reward
        self.time_step += 1
        self.rewards.append(reward)
        # Update the buffed reward memory with the bonus factor
        for arm in range(len(self.bandit.arms)):
            if self.count_memory[arm] > 0:
                self.buffed_reward_memory[arm] = (self.reward_memory[arm] / self.count_memory[arm]) + self.confidence* np.sqrt(np.log(self.time_step) / self.count_memory[arm])
            else:
                self.buffed_reward_memory[arm] = float('inf')  # force initial exploration

    
 
    def plot_arm_graph(self):
        counts = self.count_memory
        indices = np.arange(len(counts))

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.bar(indices, counts, color='skyblue', edgecolor='black')

        # Formatting
        plt.title(f'Counts per Category (confidence={self.confidence})', fontsize=16)
        plt.xlabel('Arm', fontsize=14)
        plt.ylabel('Pull Count', fontsize=14)
        plt.grid(axis='y', linestyle='-')
        plt.xticks(indices, [f'Category {i+1}' for i in indices], rotation=45, ha='right')
        # Annotate the bars with the count values
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12, color='black')
        
# # Code to test
# if __name__ == "__main__":
#     # Init Bandit
#     TIME_HORIZON = 10_000
#     bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
#     agent = UCB(TIME_HORIZON,bandit,1) ## Fill with correct constructor

#     # Loop
#     for i in range(TIME_HORIZON):
#         agent.give_pull()

#     # Plot curves
#     agent.plot_reward_vs_time_curve()
#     agent.plot_arm_graph()
#     bandit.plot_cumulative_regret()


#understanding confidence in UCB:
# The confidence parameter in UCB controls the exploration-exploitation trade-off.
TIME_HORIZON = 10_000  # Total time steps for the game
c = [0.5, 0.75, 1.0, 1.25, 2, 2.5, 3.0]
regrets = []
for conf in c:
    avg_regret = np.zeros(TIME_HORIZON)
    for _ in range(5):
        bandit = MultiArmedBandit(np.array([0.4, 0.6, 0.85, 0.15]))
        agent = UCB(TIME_HORIZON, bandit, conf)
        for i in range(TIME_HORIZON):
            agent.give_pull()
        avg_regret += bandit.cumulative_regret_array[:TIME_HORIZON].copy()  # Accumulate regrets over multiple runs
    avg_regret /= 5
    regrets.append(avg_regret)

plt.figure(figsize=(12, 6))
for idx, conf in enumerate(c):
    plt.plot(regrets[idx], label=f'c={conf}')
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret for Different Confidence Values (Averaged over 5 runs)')
plt.legend()
plt.grid()
plt.show()



# Test with a more complicated bandit (many arms, close means, some deceptive arms)
complicated_means = np.array([
    0.5, 0.51, 0.49, 0.52, 0.48, 0.53, 0.47, 0.54, 0.46, 0.55,  # Many close arms
    0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.35, 0.65,         # Some deceptive outliers
    0.56, 0.57, 0.58, 0.59, 0.6                                 # Subtle best arms
])
regrets_complicated = []
for conf in c:
    avg_regret = np.zeros(TIME_HORIZON)
    for _ in range(5):
        bandit = MultiArmedBandit(complicated_means)
        agent = UCB(TIME_HORIZON, bandit, conf)
        for i in range(TIME_HORIZON):
            agent.give_pull()
        avg_regret += bandit.cumulative_regret_array[:TIME_HORIZON].copy()
    avg_regret /= 5
    regrets_complicated.append(avg_regret)

plt.figure(figsize=(12, 6))
for idx, conf in enumerate(c):
    plt.plot(regrets_complicated[idx], label=f'c={conf}')
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret for Different Confidence Values (Complicated Bandit, Averaged over 5 runs)')
plt.legend()
plt.grid()
plt.show()

