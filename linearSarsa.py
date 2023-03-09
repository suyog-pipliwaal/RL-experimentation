from LinearWrapper import LinearWrapper
from frozenlake import FrozenLake
import numpy as np
from chooseAction import choose_action
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()

        q = features.dot(theta)
        # select an action based on epsilon greedy policy
        if random_state.rand() < epsilon[i]:
            action = random_state.choice(env.n_actions)
        else:
            qmax = max(q)
            best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
            action = random_state.choice(best)

        done = False
        while not done: # while terminal state is not reached
            next_features, reward, done = env.step(action) 
            next_q = next_features.dot(theta)
            
            # select an action based on epsilon greedy policy
            if random_state.rand() < epsilon[i]:
                next_action = random_state.choice(env.n_actions)
            else:
                qmax = max(next_q)
                best = [na for na in range(env.n_actions) if np.allclose(qmax, next_q[na])]
                next_action = random_state.choice(best)
            
            # update value of theta
            delta  = reward + gamma*next_q[next_action] - q[action]
            theta = theta + eta[i]*delta*features[action,:] 

            features = next_features # update value of q and action to move forward in game
            q = features.dot(theta)
            action = next_action
    return theta

# if __name__ =='__main__':
#     seed = 0
#     max_episodes = 2000
#     eta = 0.5
#     epsilon = 0.5
#     gamma = 0.9
#     lake =   [['&', '.', '.', '.'],
#               ['.', '#', '.', '#'],
#               ['.', '.', '.', '#'],
#               ['#', '.', '.', '$']]

#     env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
#     linear_env = LinearWrapper(env)

#     print('## Linear Sarsa')

#     parameters = linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
#     policy, value = linear_env.decode_policy(parameters)
#     linear_env.render(policy, value)

#     print('')
