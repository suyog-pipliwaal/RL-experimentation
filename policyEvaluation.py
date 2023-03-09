import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    policy_evaluation_iterations = 0
    
    # Force the loop entry
    delta = abs(theta) + 1  

    # Run the loop until conditions are reached
    while delta > theta and policy_evaluation_iterations < max_iterations:
        delta = 0
        
        # iterate every state
        for state in range(env.n_states):
            previous_value = value[state]
            total_expected_return = 0
            
            # all actions that can be taken from this state
            for next_state in range(env.n_states):
                # get the probability of transitioning to the next state
                next_state_probability = env.p(next_state, state, action=policy[state])
                
                # calculate the discounted reward
                discounted_reward = env.r(next_state, state, action=policy[state]) + (gamma * value[next_state])
                
                # calculate the expected return
                total_expected_return += next_state_probability * discounted_reward
            # new expected state
            value[state] = total_expected_return
            
            # calculate if the threshold is reached
            delta = max(delta, np.abs(previous_value - value[state]))
        
        policy_evaluation_iterations += 1

    return value