import numpy as np
def choose_action(epsilon, random_state, actions):
    if random_state.uniform(0, 1) < epsilon:
        return random_state.randint(0, 4)
    else:
        max_action = np.max(actions)
        max_index = np.flatnonzero(max_action == actions)
    return random_state.choice(max_index)
