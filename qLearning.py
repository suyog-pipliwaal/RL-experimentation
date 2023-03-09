import numpy as np
from chooseAction import choose_action

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        
        done = False

        while not done:
            #Select action a for state s according to an e-greedy policy based on Q.
            action = choose_action(epsilon[i], random_state, q[s])
            
            #Observed next state and reward for action at state
            next_state, reward, done= env.step(action)

            #Calculate the state-action value 
            q[s, action] += eta[i] * (reward + (gamma * max(q[next_state])) - q[s, action])

            #Update the state
            s = next_state
    
    #Update the policy with the maximum value
    policy = q.argmax(axis=1)
    #Update the value with the maximum value
    value = q.max(axis=1)
        
    return policy, value