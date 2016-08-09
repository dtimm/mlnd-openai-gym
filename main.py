import sys
import gym
import numpy as np
from QLAgent import QLAgent

def main():
    environment = 'CartPole-v0'
    if len(sys.argv) > 1:
        environment = sys.argv[1]

    episodes = 5001
    if len(sys.argv) > 2:
        episodes = int(sys.argv[2])

    goal = 195
    if len(sys.argv) > 3:
        goal = int(sys.argv[3])

    time = 200

    env = gym.make(environment)

    agent = QLAgent(env)
    scores = []

    for i_episode in range(episodes):
        # Get initial observation.
        agent.reset()
        observation = env.reset()

        score = 0

        # Run n = time steps
        for t in range(time):
            # Save the previous state.
            prev_state = observation

            #env.render()
            #if i_episode % 500 == 0:
            #    env.render()

            next_action = agent.get_action(observation)

            observation, reward, done, info = env.step(next_action)

            score += reward

            agent.update(prev_state, next_action, score, observation, done)

            if done or t == 199:
                print i_episode, score
                scores.append(score)

                running_avg = np.average(scores[-100:])
                #if running_avg > goal:
                #    print '100-run average {0} on run {1}!'.format(\
                #        running_avg, i_episode)

                #if i_episode % 50 == 0:
                #    print '{0} average score at {1}'.format(running_avg, \
                #        i_episode)
                                
                break
    
    return

if __name__ == "__main__":
    main()