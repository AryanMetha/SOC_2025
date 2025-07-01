import math

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import logging
import time
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
timestamp = time.strftime("%Y%m%d-%H%M%S")

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


num_envs = 16  # Run 16 environments in parallel
torch.set_num_threads(8)  

# Create vectorized environments - correct way
def make_env():
    return gym.make("LunarLander-v3", continuous=False, gravity=-10.0, 
                   enable_wind=False, wind_power=0, turbulence_power=0, render_mode=None)

envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])

STATE_SIZE = envs.single_observation_space.shape[0]
ACTION_SIZE = envs.single_action_space.n

episode_limit = 75000  # Reduced because of parallel envs
gamma_discount = 0.99 #Improve convergence properties
batch_size = 32  # M Batch updates every 32 steps

def learning_rate(step):
    return max(0.005 * np.exp(-0.02 * step / 10000), 0.001)

# Define the neural network model
class MyNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_actions_and_log_probs(states, policy):
    """Get actions and log probabilities for batch of states"""
    logits = policy(states)
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    return actions, log_probs

# Initialize networks
policy = MyNN(STATE_SIZE, ACTION_SIZE).to(device)
value_function = MyNN(STATE_SIZE, 1).to(device)

# Standard optimizers instead of eligibility traces
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
value_optimizer = torch.optim.Adam(value_function.parameters(), lr=0.001)

# Experience storage for batch updates
class ExperienceBuffer:
    def __init__(self, batch_size, num_envs, state_size, device):
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.device = device
        
        # Pre-allocate tensors
        self.states = torch.zeros(batch_size, num_envs, state_size, device=device)
        self.actions = torch.zeros(batch_size, num_envs, dtype=torch.long, device=device)
        self.rewards = torch.zeros(batch_size, num_envs, device=device)
        self.log_probs = torch.zeros(batch_size, num_envs, device=device)
        self.values = torch.zeros(batch_size, num_envs, device=device)
        self.dones = torch.zeros(batch_size, num_envs, dtype=torch.bool, device=device)
        
        self.step = 0
    
    def add(self, state, action, reward, log_prob, value, done):
        idx = self.step % self.batch_size
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.dones[idx] = done
        self.step += 1
    
    def is_full(self):
        return self.step % self.batch_size == 0 and self.step > 0
    
    def get_batch(self):
        return (self.states, self.actions, self.rewards, 
                self.log_probs, self.values, self.dones)

# Initialize experience buffer
buffer = ExperienceBuffer(batch_size, num_envs, STATE_SIZE, device)

# Tracking
episode_rewards = [[] for _ in range(num_envs)]
episode_lengths = [0] * num_envs
performance = []
total_steps = 0

# Reset environments
states, infos = envs.reset()
states = torch.FloatTensor(states).to(device)

for episode in tqdm(range(episode_limit)):
    
    # Collect batch_size steps of experience
    for step in range(batch_size):
        with torch.no_grad():
            # Get actions for all environments at once
            actions, log_probs = get_actions_and_log_probs(states, policy)
            values = value_function(states).squeeze(-1)
        
        # Step all environments
        next_states, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
        
        # Convert to tensors
        next_states = torch.FloatTensor(next_states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.BoolTensor(terminated | truncated).to(device)
        
        # Store experience
        buffer.add(states, actions, rewards, log_probs, values, dones)
        
        # episode statistics
        for i in range(num_envs):
            episode_rewards[i].append(rewards[i].item())
            episode_lengths[i] += 1
            
            if dones[i]:
                episode_reward = sum(episode_rewards[i])
                if len(performance) == 0 or len(performance) % 100 == 0:
                    logging.info(f"Episode reward: {episode_reward:.2f}, Length: {episode_lengths[i]}")
                performance.append(episode_reward)
                
                # Reset tracking for this environment
                episode_rewards[i] = []
                episode_lengths[i] = 0
        
        states = next_states
        total_steps += num_envs
        
        # Early termination check
        if len(performance) >= 100:
            recent_avg = np.mean(performance[-100:])
            if recent_avg >= 200:
                logging.info(f"EARLY COMPLETION! Average reward: {recent_avg:.2f}")
                break
    
    # Update networks when buffer is full
    if buffer.is_full():
        batch_states, batch_actions, batch_rewards, batch_log_probs, batch_values, batch_dones = buffer.get_batch()
        
        # Compute advantages using TD error
        with torch.no_grad():
            # Get next values for advantage computation
            next_values = value_function(states).squeeze(-1)
            
            advantages = torch.zeros_like(batch_rewards)
            returns = torch.zeros_like(batch_rewards)
            
            # Compute advantages and returns
            for t in range(batch_size):
                if t == batch_size - 1:
                    # Use current next_values for last step
                    next_value = torch.where(batch_dones[t], torch.zeros_like(next_values), next_values)
                else:
                    # Use next step's value
                    next_value = torch.where(batch_dones[t], torch.zeros_like(batch_values[t+1]), batch_values[t+1])
                
                td_target = batch_rewards[t] + gamma_discount * next_value
                advantages[t] = td_target - batch_values[t]
                returns[t] = td_target
        
        # Flatten for batch processing
        flat_states = batch_states.view(-1, STATE_SIZE)
        flat_actions = batch_actions.view(-1)
        flat_log_probs = batch_log_probs.view(-1)
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)
        
        # Update policy
        policy_optimizer.zero_grad()
        
        new_logits = policy(flat_states)
        new_dist = torch.distributions.Categorical(logits=new_logits)
        new_log_probs = new_dist.log_prob(flat_actions)
        
        # Policy loss (REINFORCE with baseline)
        policy_loss = -(new_log_probs * flat_advantages).mean()
        
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy_optimizer.step()
        
        # Update value function
        value_optimizer.zero_grad()
        
        new_values = value_function(flat_states).squeeze(-1)
        value_loss = F.mse_loss(new_values, flat_returns)
        
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_function.parameters(), 0.5)
        value_optimizer.step()
    
    # Check early termination
    if len(performance) >= 500:
        recent_avg = np.mean(performance[-500:])
        if recent_avg >= 200:
            break

envs.close()

# Save model
torch.save({
    'policy_state_dict': policy.state_dict(),
    'value_state_dict': value_function.state_dict(),
    'performance': performance,
    'total_steps': total_steps
}, f'vectorized_checkpoint_{timestamp}.pt')

# Plot results
if len(performance) > 0:
    # Smooth the performance curve
    window_size = min(50, len(performance) // 10)
    if window_size > 1:
        smoothed = np.convolve(performance, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed, label=f'Smoothed (window={window_size})')
    
    plt.plot(performance, alpha=0.3, label='Raw episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.title('Training Performance')
    plt.legend()
    plt.grid(True)
    plt.show()



print(f"Training completed! Total steps: {total_steps}")
print(f"Final average reward: {np.mean(performance[-100:]) if len(performance) >= 100 else 'N/A'}")