import random
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def state_from_observation(observation):
    state = ()
    for val in observation:
        temp_value = round(val * 8.0)/8.0
        if temp_value == -0.0:
            temp_value = 0.0
        state += (temp_value, )
    return state


class QLAgent:
    def __init__(self, env, time):
        self.env = env
        self.q_states = {}
        self.q_experience = {}
        self.alpha = 0.5
        self.gamma = 0.95
        self.time = time
        self.timeleft = 200
        self.random_act = 1.0
        self.random_decay = 0.997

    def reset(self):
        self.timeleft = self.time
        self.random_act *= self.random_decay
        return

    def get_action(self, obs):
        state = state_from_observation(obs)

        if state in self.q_states.keys():
            # Take the best scored available action.
            action = max(self.q_states[state], key=self.q_states[state].get)
            
        else:
            # Take a random actions if you've never felt like this before.
            action = self.env.action_space.sample()

            # Initialize this new state.
            self.q_states[state] = {}

            for act in xrange(self.env.action_space.n):
                self.q_states[state][act] = 0.
        
        # Chance of random action.
        if random.random() < self.random_act:
            action = self.env.action_space.sample()

        return action
    
    def update(self, obs, action, reward, obs_prime, done):
        self.timeleft -= 1

        state = state_from_observation(obs)
        state_prime = state_from_observation(obs_prime)

        next_best = 0.0
        if state_prime in self.q_states.keys():
            next_best = max(self.q_states[state_prime].values())

        if done == True:
            if reward < 0:
                # Ended the pain!
                reward = 50.
            else:
                reward = -50.
        
        self.q_states[state][action] = (1.0 - self.alpha) * \
                            self.q_states[state][action] + \
                            self.alpha * (reward + next_best)
        return