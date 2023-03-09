import numpy as np
from environment import Environment
import contextlib

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

class FrozenLake(Environment):

    def __init__(self, lake, slip, max_steps, seed=None):

        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        self.action_probabilities = np.load('p.npy')

        self.width = len(lake[0])
        self.height = len(lake)

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)

        # TODO:

    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    #Method P returns the probability of going from one state to another
    def p(self, next_state, state, action):
        # TODO:

        
        possible_adjacent_states = []
        distributional_probabilities = np.zeros(self.n_states)

        for i in range(self.n_actions):
            pas = self.take_action(state, i)
            possible_adjacent_states.append(pas)

        shared_slip_probability = self.slip/len(possible_adjacent_states)

        for adjacent_state in possible_adjacent_states:
            distributional_probabilities[adjacent_state] += shared_slip_probability

        next_expected_state = self.take_action(state, action)
        distributional_probabilities[next_expected_state] += (1 - self.slip)

        return distributional_probabilities[next_state]


    #The method r returns the expected reward in having transitioned from state to next state given action
    def r(self, next_state, state, action):
        # TODO:

        if self.goal_state(state):
           return 1

        return 0

    #NEW METHOD take_action returns the coordinates of the new state after taking an action
    def take_action(self, state, action):

        if state == self.absorbing_state:
            return state

        #if in the hole state or goal state enter the absorbing state
        if self.hole_state(state) or self.goal_state(state):
            return self.absorbing_state

        state_coordinates = self.state_to_coordinates(state)
        action_coordinates = self.action_to_coordinates(action)

        #transitions to next state
        transition_state_coordinates = [

            state_coordinates[0] + action_coordinates[0],
            state_coordinates[1] + action_coordinates[1]

        ]

        next_state = self.coordinates_to_state(transition_state_coordinates)
        return int(next_state) if self.valid_coordinates(transition_state_coordinates) else int(state)

    #NEW METHOD that avoids picking cooridinates that are out of environment
    def valid_coordinates(self, coordinates):

        if (coordinates[0] < 0) or (coordinates[0] >= self.width):
            return False

        if (coordinates[1] < 0) or (coordinates[1] >= self.height):
            return False

        return True

    #NEW METHOD method defines the hole state
    def hole_state(self, state):
        if state == self.absorbing_state:
            return False

        return self.lake_flat[int(state)] == '#'

    #NEW METHOD this method defines the goal
    def goal_state(self, state):
        if state == self.absorbing_state:
            return False

        return self.lake_flat[int(state)] == '$'

    #NEW METHOD transfroms the states to coordinates on the environment
    def state_to_coordinates(self,state):

        x_index = state % self.width
        y_index = (state - x_index) / self.width
        return [x_index, y_index]

    #NEW METHOD transfroms the states to coordinates on the environment
    def coordinates_to_state(self, coordinates):
        return (coordinates[1] * self.width) + coordinates[0]

    #NEW METHOD transfroms the actions to coordinates on the environment
    def action_to_coordinates(self, action):

        if action == 0: #UP
           return [0, -1]

        if action == 1: #LEFT
           return [-1, 0]

        if action == 2: #DOWN
           return [0, 1]

        if action == 3: #RIGHT
           return [1, 0]

        return [0, 0]



    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

def play(env):
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}, done:{1}.'.format(r, done))

# seed = 0

# # Small lake
# lake = [['&', '.', '.', '.'],
#         ['.', '#', '.', '#'],
#         ['.', '.', '.', '#'],
#         ['#', '.', '.', '$']]

# env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

# play(env)
