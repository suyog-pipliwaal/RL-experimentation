import numpy as np
from frozenlake import FrozenLake
import numpy as np
from chooseAction import choose_action
def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    # eta is the learning rate decay linearly eta[i] is the learning rate for episode i

    epsilon = np.linspace(epsilon, 0, max_episodes)
    # epsilon is decay linearly espilon[i] is the for episode i

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        action = choose_action(epsilon[i], random_state, q[s])
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = choose_action(epsilon[i], random_state, q[s])
            q[s, action] += eta[i]* (reward + (gamma * q[next_state, next_action]-q[s, action]))
            s = next_state
            action = next_action


    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value