import matplotlib.pyplot as plt
import numpy as np

discount_factor=1
value_estimates = np.zeros(101) 
value_estimates[100] = 1  # terminal state value
policy = np.ones(100, dtype=int)  # initial policy: bet 1, ensure integer type
policy[0] = 0  # state 0 is terminal, no bet

p_h=0.7

policy_sweeps=50  #assume that in 10 sweeps you converge to the accurate value function for that policy(is this even necessary or is just one sweep enough ??)
iterations=1000 # assume that in 10 such iterations you reach the optimal policy 


for iteration in range(iterations):
    #step 1 :evaluation of policy:inital policy is to bet 
    for sweep in range(policy_sweeps):
        for state in range(1, 100):
            if state==0:
                value_estimates[state] = 0
            else:
                if state + policy[state] >= 100:
                    value_estimates[state] = p_h * (value_estimates[100]) + (1 - p_h) * value_estimates[state-policy[state]]
                else:
                    value_estimates[state] = p_h * value_estimates[state+policy[state]] + (1 - p_h) * value_estimates[state-policy[state]] 

    #step 2: new policy = greedy (value function with previous policy)
    for state in range(1, 100):
        valid_bets=[ bet for bet in range(1,min(100-state, state)+1)]
        action_values_q=np.zeros(len(valid_bets))
        for bet in valid_bets:
            if state + bet >= 100:
                #action_values_q[bet] = p_h * (1 + value_estimates[100]) + (1 - p_h) * value_estimates[state - bet]
                action_values_q[bet-1] = p_h * (1) + (1 - p_h) * value_estimates[state - bet]
            else:
                action_values_q[bet-1]= p_h*(discount_factor*value_estimates[state+bet])+(1-p_h)*(discount_factor*value_estimates[state-bet])

        policy[state]=np.argmax(action_values_q)+1

    if iteration in [1,2,3,10,20,30]:
        #draw graph of value function (on a single plot)
        # Store value estimates for plotting after the loop
        if 'value_estimates_history' not in locals():
            value_estimates_history = []
            iteration_labels = []
        value_estimates_history.append(value_estimates.copy())
        iteration_labels.append(f'Iteration {iteration}')
    

# Plotting the value function for each iteration
for estimates, label in zip(value_estimates_history, iteration_labels):
    plt.plot(range(101), estimates, label=label)
plt.xlabel('State')
plt.ylabel('Value')
plt.title('Gambler\'s Problem Value Function Over Iterations')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(0, 101, 10))
plt.show()


#plot final policy
plt.plot(range(100), policy, label='Policy', linestyle='--')
plt.xlabel('State')
plt.ylabel('final Policy')
plt.title('Gambler\'s Problem Policy')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(0, 101, 10))
print("Final Policy:", policy)
plt.show()



   
    