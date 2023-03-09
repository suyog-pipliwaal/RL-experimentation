import numpy as np
from frozenlake import FrozenLake

def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    #Initialize policy with zeros
    policy = np.zeros(env.n_states, dtype=int)

    n_value_iterations = 0
    delta = abs(theta) + 1 

    while max_iterations > n_value_iterations and delta > theta:
        delta = 0

        for state in range(env.n_states):
            previous_value = value[state]
            new_value = []

            for action in range(env.n_actions):
                total_exprected_return = 0

                for next_state in range(env.n_states):
                    #Probability of transitioning to this state to the next state
                    next_state_probability = env.p(next_state, state, action)

                    #Get the discounted reward
                    discounted_reward = env.r(next_state, state, action) + (gamma*value[next_state])
                    # print("-------", discounted_reward)

                    #Expected return for each state
                    total_exprected_return += next_state_probability * discounted_reward
                
                new_value.append(total_exprected_return) #Append the expected return for each next state

            #Select the maximum value
            value[state] = max(new_value)

            #Calculate how much the estimate of the value for this state has changed and store the maximum from change we've already observed
            delta = max(delta, np.abs(previous_value - value[state]))
        
        n_value_iterations += 1

    # Recover the best action of the optimal policy
    for state in range(env.n_states):
        new_actions = []
        new_action_values = []
        for action in range(env.n_actions):
            for next_state in range(env.n_states):
                #Probability of transitioning to this state to the next state
                next_state_probability = env.p(next_state, state, action=action)
                
                #Get the discounted reward
                discounted_reward = env.r(next_state, state, action=action) + (gamma*value[next_state])

                #Appned the new actions
                new_actions.append(action)
                #Append the value to the new action
                new_action_values.append(next_state_probability*discounted_reward)

        #Choose the maximum value as it will give the best optimal action
        best_action = new_actions[new_action_values.index(max(new_action_values))]
        policy[state] = best_action

    print("Number of value iterations :-> ",n_value_iterations)

    return policy, value
