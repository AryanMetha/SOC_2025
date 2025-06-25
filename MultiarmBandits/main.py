# Write code which will run all the different bandit agents together and:
# 1. Plot a common cumulative regret curves graph
# 2. Plot a common graph of average reward curves

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from base import MultiArmedBandit
from ucb import UCB
from thompson import ThompsonAgent
from klucb import KLUCBAgent
from epsilon_greedy import EpsilonGreedyAgent
from tqdm import tqdm

game_time = 30000  # Total time steps for the game=30k


def game_one(ucb_confidence=1.0, e=0.1):
    agents = [
        UCB(game_time,MultiArmedBandit(np.array([0.1, 0.2, 0.3, 0.4, 0.5])),ucb_confidence),    # UCB agent with confidence level 1.0
        ThompsonAgent(game_time,MultiArmedBandit(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))),           # Thompson sampling agent
        KLUCBAgent(game_time, MultiArmedBandit(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))),                # KL-UCB agent
        EpsilonGreedyAgent(game_time, MultiArmedBandit(np.array([0.1, 0.2, 0.3, 0.4, 0.5])),e)  # Epsilon-greedy agent with epsilon=0.1
    ]

    for agent in tqdm(agents, desc="Running agents"):
        for _ in range(game_time):
            agent.give_pull()
    

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for idx, agent in enumerate(agents):
        ax = axes[idx]
        regret = agent.bandit.cumulative_regret_array
        time_steps = np.arange(len(regret))

        # Dynamic title with epsilon if applicable
        title = agent.__class__.__name__
        if hasattr(agent, 'epsilon'):
            title += f' (ε = {agent.epsilon})'

        # Plot the regret curve
        ax.plot(time_steps, regret, color=f'C{idx}', linewidth=2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Cumulative Regret', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Shared x-axis range: 0–30,000
        ax.set_xlim(0, 30000)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x / 1000)}k'))

        # Format y-ticks for readability
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y):,}'))

        # Set y-limits based on agent type for better visibility
        ax.tick_params(axis='both', which='major', labelsize=10)
        if isinstance(agent, EpsilonGreedyAgent):
            ax.set_ylim(0, 700)
        elif isinstance(agent, KLUCBAgent):
            ax.set_ylim(-20, 180)
        elif isinstance(agent, ThompsonAgent):
            ax.set_ylim(-30, 200)
        else:  # UCB agent
            ax.set_ylim(-10, 300)
        
        # Plot average reward curve on the same subplot (secondary y-axis)
        avg_reward = np.cumsum(agent.bandit.reward_array) / (np.arange(1, len(agent.bandit.reward_array) + 1))
        ax2 = ax.twinx()
        ax2.plot(time_steps, avg_reward, color=f'C{idx}', linestyle='--', alpha=0.7, label='Avg Reward')
        ax2.set_ylabel('Average Reward', fontsize=12)
        ax2.tick_params(axis='y', labelsize=10)
        ax2.set_ylim(0, 0.6)

        # --- Cumulative Reward Plot (shared for all agents) ---
        if idx == len(agents) - 1:
            fig2, ax_cumrew = plt.subplots(figsize=(10, 6))
            for j, ag in enumerate(agents):
                cum_reward = np.cumsum(ag.bandit.reward_array)
                ax_cumrew.plot(np.arange(len(cum_reward)), cum_reward, label=ag.__class__.__name__, linewidth=2)
            ax_cumrew.set_title('Cumulative Reward (All Agents)', fontsize=16)
            ax_cumrew.set_xlabel('Time Steps', fontsize=12)
            ax_cumrew.set_ylabel('Cumulative Reward', fontsize=12)
            ax_cumrew.legend(fontsize=12)
            ax_cumrew.grid(True, linestyle='--', alpha=0.5)
            ax_cumrew.set_xlim(0, 30000)
            ax_cumrew.xaxis.set_major_locator(ticker.MultipleLocator(5000))
            ax_cumrew.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x / 1000)}k'))
            plt.tight_layout()
            plt.show()

        # --- Arm Selection Plot (plot_arm_graph for each agent) ---
        if idx == len(agents) - 1:
            fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
            axes3 = axes3.flatten()
            for k, ag in enumerate(agents):
                plt.sca(axes3[k])
                ag.plot_arm_graph()
                axes3[k].set_title(f"{ag.__class__.__name__} Arm Selection", fontsize=14)
            fig3.suptitle('Arm Selection Over Time', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    # Overall layout
    fig.suptitle('Cumulative Regret (Individual Scales)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


#game_one()

def game_two(ucb_confidence=1.0, e=0.1):
    p_values= [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    final_regrets = [[0,0,0,0] for _ in range(len(p_values))]  # Store final regrets for each agent
    for i, p in enumerate(p_values):
        print(f"Running game with p = {p}")
        agents = [
            UCB(game_time, MultiArmedBandit(np.array([p,  p+0.1])), ucb_confidence),
            ThompsonAgent(game_time, MultiArmedBandit(np.array([p,  p+0.1]))),
            KLUCBAgent(game_time, MultiArmedBandit(np.array([p,  p+0.1]))),
            EpsilonGreedyAgent(game_time, MultiArmedBandit(np.array([p,  p+0.1])), e)
        ]

        #just to get a better understanding of the agents:
        agents = [
            UCB(game_time, MultiArmedBandit(np.array([p,  p+0.1,p+0.05,p-0.05])), ucb_confidence),
            ThompsonAgent(game_time, MultiArmedBandit(np.array([p,  p+0.1,p+0.05,p-0.05]))),
            KLUCBAgent(game_time, MultiArmedBandit(np.array([p,  p+0.1,p+0.05,p-0.05]))),
            EpsilonGreedyAgent(game_time, MultiArmedBandit(np.array([p,  p+0.1,p+0.05,p-0.05])), e)
        ]

        num_iterations = 100
        for iteration in tqdm(range(num_iterations), desc=f"Iterations for p={p}"):
            # Re-initialize agents for each iteration to reset their state
            agents_iter = [
            UCB(game_time, MultiArmedBandit(np.array([p,  p+0.1,p+0.05,p-0.05])), ucb_confidence),
            ThompsonAgent(game_time, MultiArmedBandit(np.array([p,  p+0.1,p+0.05,p-0.05]))),
            KLUCBAgent(game_time, MultiArmedBandit(np.array([p,  p+0.1,p+0.05,p-0.05]))),
            EpsilonGreedyAgent(game_time, MultiArmedBandit(np.array([p,  p+0.1,p+0.05,p-0.05])), e)
            ]
            for agent in agents_iter:
                for _ in range(game_time):
                    agent.give_pull()
            for j, agent in enumerate(agents_iter):
                final_regrets[i][j] += agent.bandit.cumulative_regret_array[-1]
        # Average regrets over iterations
        for j in range(4):
            final_regrets[i][j] /= num_iterations

    # Plotting the final regrets for each agent across different p values
    fig, ax = plt.subplots(figsize=(12, 6))
    for j, agent_name in enumerate(['UCB', 'Thompson', 'KL-UCB', 'Epsilon-Greedy']):
        regrets = [final_regrets[i][j] for i in range(len(p_values))]
        ax.plot(p_values, regrets, marker='o', label=agent_name)
    ax.set_title('Final Regrets for Different Agents Across Varying p Values', fontsize=16)
    ax.set_xlabel('p Value', fontsize=14)
    ax.set_ylabel('Final Cumulative Regret', fontsize=14)
    ax.legend(title='Agents', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(p_values)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))
    plt.tight_layout()
    plt.show()



game_two()
