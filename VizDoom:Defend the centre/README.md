# VizDoom for RL

## Task 5: Train an agent for Defend the Centre
Several Key learnings in this section. Few highlights are
* Mathematical/Conceptual understanding of TRPO and PPO
* Implementation using Stable Baselines-key practices
* Reward Shaping and curriculum learning


I achiveved a much better performace than what was demonstrated in the sample by making the following changes

1. Increased the number of training steps to allow the agent more time to learn.
2. Tuned the learning rate and batch size for more stable training.
3. Applied reward shaping to encourage survival and accurate shooting.(by giving reward for survival and killing enimies and punishing for taking a shot and taking damage)
4. used a vectorized environment running 4 envs simultanouesly 

* Another learning : since i had tied reward to ep length , in call back i shd have included something analogous to kill count(total reward-reward fo surviving) to also montior that (although the increase in it with rise in episode length is implicit).
* And also leraning rate was quite low , it could have been higher

![Reward,loss](imgs/Screenshot%20from%202025-07-06%2017-30-43.png)
![Final Metrics](imgs/Screenshot%20from%202025-07-06%2017-31-00.png)

