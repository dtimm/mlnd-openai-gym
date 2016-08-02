import random
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def state_from_observation(observation):
    state = ()
    for val in observation:
        temp_value = round(val * 4.0)/4.0
        if temp_value == -0.0:
            temp_value = 0.0
        state += (temp_value, )
    return state


class QLAgent:
    def __init__(self, env):
        self.env = env
        self.q_states = {}
        self.alpha = 0.5
        self.gamma = 0.95

    def state_from_observation(observation):
        state = ()
        for val in observation:
            temp_value = round(val * 4.0)/4.0
            if temp_value == -0.0:
                temp_value = 0.0
            state += (temp_value, )

        return state

    def get_action(self, obs):
        state = state_from_observation(obs)

        if state in self.q_states.keys():
            # default action is random if nothing is better.
            values = softmax(self.q_states[state].values())
            if random.random() < values[0]:
                action = 0
            else:
                action = 1
            
        else:
            # Find the most similar state
            best_diff = -1.0
            best_state = None
            for s in self.q_states.keys():
                i = 0
                distance = 0
                for val in s:
                    distance += (val - s[i])**2
                
                if distance < best_diff or best_state == None:
                    best_diff = distance
                    best_state = s
            
            if best_state != None and best_diff < 0.5:
                self.q_states[state] = self.q_states[best_state].copy()
                action = self.env.action_space.sample()
            else:
                # Take a random actions if you've never felt like this before.
                action = self.env.action_space.sample()

                # Initialize this new state.
                self.q_states[state] = {}

                for act in xrange(self.env.action_space.n):
                    self.q_states[state][act] = 1
        
        return action
        return self.env.action_space.sample()
    
    def update(self, obs, action, reward, obs_prime):
        state = state_from_observation(obs)
        state_prime = state_from_observation(obs_prime)

        next_best = 1.0
        if state_prime in self.q_states.keys():
            next_best = max(self.q_states[state_prime].values())
        
        self.q_states[state][action] = (1.0 - self.alpha) * self.q_states[state][action] + \
                            self.alpha * (reward + self.gamma * next_best)
        return