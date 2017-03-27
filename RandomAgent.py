
class RandomAgent:
    def __init__(self, env):
        self.env = env
    
    def reset(self):
        return

    def get_action(self, state, report=False):
        action = self.env.action_space.sample()
        if report:
            print(action)
        return action
    
    def update(self, state, action, reward, state_prime, done):
        return