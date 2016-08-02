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
    def __init__(self, env, time):
        self.env = env
        self.q_states = {}
        self.q_experience = {}
        self.alpha = 0.2
        self.gamma = 0.9
        self.time = time
        self.timeleft = 200

    def reset(self):
        self.timeleft = self.time
        return

    def get_cumulative_reward(self, state, action):
        i = self.timeleft
        c_gamma = 1.0
        c_reward = 0.0
        next_action = action

        c_reward += self.q_states[state][action]

        while i > 0:
            i -= 1
            c_gamma *= self.gamma

            if state not in self.q_states.keys():
                i = 0
                break

            c_reward += c_gamma * self.q_states[state][next_action]

            if state in self.q_experience.keys():
                if next_action in self.q_experience[state].keys():
                    state = self.q_experience[state][next_action]
                    if state in self.q_states.keys():
                        next_action = max(self.q_states[state], key=self.q_states[state].get)
                    else:
                        i = 0
                        break
                else:
                    i = 0
                    break
            else:
                i = 0
                break

        return c_reward

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
        self.timeleft -= 1

        state = state_from_observation(obs)
        state_prime = state_from_observation(obs_prime)

        if state not in self.q_experience.keys():
            self.q_experience[state] = {}
        self.q_experience[state][action] = state_prime

        next_best = self.get_cumulative_reward(state, action)
        
        if next_best < -1.0:
            next_best = -1.0
        if next_best > 1.0:
            next_best = 1.0
        
        self.q_states[state][action] = (1.0 - self.alpha) * self.q_states[state][action] + \
                            self.alpha * (reward + next_best)
        return