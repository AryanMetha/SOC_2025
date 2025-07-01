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

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,enable_wind=False, wind_power=0, turbulence_power=0, render_mode=None)
env.reset()

STATE_SIZE = env.observation_space.shape[0]  # Number of state features
ACTION_SIZE = env.action_space.n  # Number of discrete actions

episode_limit=1e7
gamma_discount=1
memory_lambda=-0.7
def learning_rate(episode,decay=0.02):
    return max(0.005*np.exp(-2*episode),0.001)


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
        return x  #will return logits for each action

def get_action(state, policy):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert to tensor and add batch dimension
    logits = policy(state_tensor)  # Get action logits
    action_probs = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
    action = torch.multinomial(action_probs, num_samples=1).item()  # Sample action from the distribution
    return action

# Initialize the policy and value function networks , we do not define a score function for the policy , rather just directly use the NN logits
policy = MyNN(STATE_SIZE, ACTION_SIZE)
value_function = MyNN(STATE_SIZE, 1)
eligibility_traces = [
    torch.zeros_like(param, dtype=torch.float32) #Create a new tensor with the same shape as param, filled with zeros
    for param in policy.parameters()             #loop through all parameters , wew have 6: 3 rank2 weights and 3 rank1 biases connecting each layer
]


#not using an optimizer , manual update

#tracking tools
avg=0 #avg_reward per 500 episodes
performance=[]

for episode in tqdm(range(int(episode_limit))):
    cumulative_reward=0
    current_state,info=env.reset()
    action=get_action(current_state,policy)
    while True:
        next_state, reward, terminated, truncated, info = env.step(action)

        target = reward + gamma_discount * value_function(torch.FloatTensor(next_state).unsqueeze(0)).detach()
        predicted = value_function(torch.FloatTensor(current_state).unsqueeze(0))

        td_loss= target-predicted

        cumulative_reward+=reward

        # step 1:  Setup gradient
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        logits = policy(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        policy.zero_grad()           # clear any previous gradients
        log_prob.backward()  


        # Step 2: Update eligibility traces
        for trace, param in zip(eligibility_traces, policy.parameters()):
            if param.grad is not None:
                grad = param.grad
                if grad.shape != param.shape:
                    grad = grad.view_as(param)
                trace.mul_(gamma_discount * memory_lambda)
                trace.add_(grad)

        #step 3: make changes
        with torch.no_grad():
            for param, trace in zip(policy.parameters(), eligibility_traces):
                # Remove extra dimensions (like [1, 128] → [128])
                trace = trace.squeeze()

                # Ensure final shape matches exactly
                if trace.shape != param.shape:
                    trace = trace.view_as(param)

                param.add_(learning_rate(episode) * td_loss * trace)


        #step 4 :update value function
        value_loss = F.mse_loss(predicted, target)
        value_function.zero_grad()
        value_loss.backward()
        with torch.no_grad():
            for param, trace in zip(policy.parameters(), eligibility_traces):
                param.add_(learning_rate(episode) * td_loss * trace)



        current_state=next_state
        action=get_action(current_state,policy)
        if terminated or truncated: break

    avg+=cumulative_reward*0.002
    if (episode+1)%500==0:
        logging.info(f"Steps Completed:{int(episode/500)+1} : Avg performance: {avg}")
        if avg >= 200*500:
            logging.info(f"EARLY COMPLETION !!!Steps Completed:{int(episode/500)+1} : Avg performance: {avg}")
            performance.append(avg)
            break
        performance.append(avg)
        avg=0

        

plt.plot(performance)
plt.xlabel('Episodes (x500)')
plt.ylabel('Average Reward')
plt.title('Performance over Time')
plt.grid(True)
plt.show()

torch.save({
    'episode': episode,
    'policy_state_dict': policy.state_dict(),
    'value_state_dict': value_function.state_dict(),
    'eligibility_traces': eligibility_traces,  
}, f'checkpoint_{timestamp}.pt')

