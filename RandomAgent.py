
class RandomAgent:
    def __init__(self, env):
        self.env = env
    
    def reset(self):
        return

    def get_action(self, state):
        return self.env.action_space.sample()
    
    def update(self, state, action, reward, state_prime):
        return