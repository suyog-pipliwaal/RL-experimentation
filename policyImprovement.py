import numpy as np

def policy_improvement(env, policy, value, gamma):
    # initialize with zeros
    improved_policy = np.zeros(env.n_states, dtype=int)
    
    #  keep track of whether the policy has improved or not
    policy_stable = True

    # iterate over every state
    for state in range(env.n_states):
        previous_action = policy[state]
        new_actions = []
        new_action_values = []

        # for every action taken, calculate the expected value of the next state given this action
        for action in range(env.n_actions):
            for next_state in range(env.n_states):
                new_actions.append(action)
                
                # probability of transitioning to the next state
                next_state_probability = env.p(next_state, state, action=action)
                
                # calculate the discounted reward
                discounted_reward = env.r(next_state, state, action=action) + (gamma * value[next_state])
                new_action_values.append(next_state_probability * discounted_reward)

        # select the action that leads to maximum reward
        best_action = new_actions[new_action_values.index(max(new_action_values))]
        improved_policy[state] = best_action

        # policy is still not stable if there is a change i.e. we need to keep improving
        if previous_action != best_action:
            policy_stable = False #if the policy is not stable

    return improved_policy, policy_stable