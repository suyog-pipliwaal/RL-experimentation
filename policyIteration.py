import numpy as np
from policyImprovement import policy_improvement
from policyEvaluation import policy_evaluation

def policy_iteration(env, gamma, theta, max_iteration, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int) 

    n_policy_iteration = 0

    while True:
        # evaluate the policy
        value = policy_evaluation(env, policy, gamma, theta, max_iteration)
        # improvement the policy
        policy, policy_stable = policy_improvement(env, policy, value, gamma)
        
        n_policy_iteration += 1
        
        # if there is no further change in the policy, the policy_stable will be True 
        if policy_stable:
            break

    print("Number of policy iterations needed:", n_policy_iteration)

    return policy, value