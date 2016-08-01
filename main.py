import random
import sys
import gym
from RandomAgent import RandomAgent

def main():
    environment = 'CartPole-v0'
    if len(sys.argv) > 1:
        environment = sys.argv[1]

    episodes = 5000
    if len(sys.argv) > 2:
        episodes = int(sys.argv[2])

    env = gym.make(environment)

    agent = RandomAgent(env)

    for i_episode in range(episodes):
        # Get initial observation.
        observation = env.reset()

        score = 0

        # Run 200 time steps
        for t in range(200):
            # Save the previous state.
            prev_state = observation

            if i_episode % 100 == 0:
                env.render()

            next_action = agent.get_action(observation)

            observation, reward, done, info = env.step(next_action)
            score += reward

            agent.update(prev_state, next_action, reward, observation)

            if done or t == 199:
                print score
                break
    return

if __name__ == "__main__":
    main()