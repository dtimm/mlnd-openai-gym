import sys
import gym
import numpy as np
import argparse
from NAFAgent import NAFAgent
from QLAgent import QLAgent
from RandomAgent import RandomAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, default='CartPole-v0',
        help='OpenAI Gym environment to run.')
    parser.add_argument('-p', '--episodes', type=int, default=1000,
        help='Number of episodes to simulate.')
    parser.add_argument('-g', '--goal', type=int, default=195,
        help='Goal score for the environment.')
    parser.add_argument('-t', '--time', type=int, default=200,
        help='Time steps for each episode.')
    parser.add_argument('-a', '--agent', type=str, default='QL',
        help='Learning agent type (QL or NAF).')
    parser.add_argument('--report', action='store_true', help='Report results.')
    parser.add_argument('-d', '--debug', action='store_true', 
        help='Print max values at each time-step.')

    args = parser.parse_args()
    print args

    environment = args.environment
    episodes = args.episodes
    goal = args.goal
    time = args.time

    env = gym.make(environment)

    if (args.agent == 'QL'):
        agent = QLAgent(env)
    elif (args.agent == 'NAF'):
        agent = NAFAgent(env)
    elif (args.agent == 'Random'):
        agent = RandomAgent(env)

    scores = []

    if args.report:
        filename = 'tmp/gym-report'
        env.monitor.start(filename, force=True)

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
            report = args.debug
            next_action = agent.get_action(observation, report)

            observation, reward, done, info = env.step(next_action)

            if report:
                print reward

            score += reward

            agent.update(prev_state, next_action, reward, observation, done)

            if done or t == time - 1:
                print i_episode + 1, score
                scores.append(score)

                running_avg = np.average(scores[-100:])
                #if running_avg > goal:
                #    print '100-run average {0} on run {1}!'.format(\
                #        running_avg, i_episode)

                if (i_episode + 1) % 50 == 0:
                    print '{0} average score at {1}'.format(running_avg, \
                        i_episode + 1)
                                
                break

    if args.report:
        env.monitor.close()
        key_file = open('api.key', 'r')
        gym_key = key_file.readline()
        if args.agent == 'NAF':
            algo_id = 'alg_xjVArtUxQXqfSq5q89dRjQ'
        else:
            algo_id = 'alg_sbIxfyjIRUSBrBA1IOFg'
        gym.upload(filename, api_key=gym_key, algorithm_id=algo_id)

    return

if __name__ == "__main__":
    main()