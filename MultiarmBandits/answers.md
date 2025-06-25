# Justification for Final Regrets

This document provides a detailed justification for the final regrets associated with the project.
## Figure 1

![Figure 1](Figure_1.png)

#### Note:
Epsilon-Greedy is significantly worse especially in case of a 2-armed bandit as much exploration is unnecessary.

## General Trend
The spike in the middle can be justified as follows :

Being a case of the classic bernaulli bandit , the highest randomness in the reward obtained will be around p~0.5 (mathematically the std deviation will be the highest at p=0.5)

As a result there is a certain "confusion" and hence more exploration is required to gain surety.


According to me the remaining spikes are random and would even out if a large number of games are conducnted , eventually leading to a sort of normal distribution(central limit theorm ????).
