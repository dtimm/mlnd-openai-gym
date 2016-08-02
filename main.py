import sys
import gym
import numpy as np
from QLAgent import QLAgent

def main():
    environment = 'CartPole-v0'
    if len(sys.argv) > 1:
        environment = sys.argv[1]

    episodes = 50000
    if len(sys.argv) > 2:
        episodes = int(sys.argv[2])

    env = gym.make(environment)

    agent = QLAgent(env)
    scores = []

    for i_episode in range(episodes):
        # Get initial observation.
        observation = env.reset()

        score = 0

        # Run 200 time steps
        for t in range(200):
            # Save the previous state.
            prev_state = observation

            if i_episode % 500 == 0:
                env.render()

            next_action = agent.get_action(observation)

            observation, reward, done, info = env.step(next_action)

            score += reward

            agent.update(prev_state, next_action, reward, observation)

            if done or t == 199:
                while len(scores) >= 100:
                    scores.pop(0)

                scores.append(score)

                if (i_episode + 1) % 100 == 0:
                    print '{0} average score at {1}'.format(np.average(scores), i_episode + 1)
                
                break
    
    return

if __name__ == "__main__":
    main()