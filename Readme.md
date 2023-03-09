# Reinforcement Learning 
In this project, we have implemented various **RL** algorithms for the given environment. We expirenmented with Frozen lake enviroment to find the optimal policies using tabular and non-tabular menthods. 


* Tabular methods refer to problems with state action pairs small enough to approximate value functions with tables or arrays which were used to implement policy evaluation, policy improvement, policy iteration, and value iteration. 
* Non-tabular methods were used in the implementation of Sarsa control and Q-learning control using linear function approximation.

## Results
* For the policy iteration to find the optimal policy for the big frozen lake, the number of iterations it required are 6. Whereas, for the value iteration the number of iterations required are 20. 
* Time taken by the policy iteration is 14.26 sec and time taken by the value iteration is 10.75 sec. Since the value iteration algorithm required less time, it was faster as compared to the policy iteration algorithm.
* Under the same commom conditions, both non tabular fucntion converge to the real value functions but at different rates. Sarsa contol require 1100 episodes to find the optimal policy whereas Q-learning control needed 3000 episodes to find the optimal policy for the the small frozen lake. 
