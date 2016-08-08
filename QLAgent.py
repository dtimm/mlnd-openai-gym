import random

import numpy as np

import gym
import tensorflow as tf
from tensorflow.contrib.framework import get_variables
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import convolution2d


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

class QLAgent:
    def __init__(self, env):
        # Initialive discounts, networks, EVERYTHING!
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.98

        self.alpha = 0.005

        self.update_samples = 200
        self.update_steps = 5
        self.env = env
        self.tf_sess = tf.Session()
        
        # Get state and action counts.
        self.states = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            #self.actions = self.env.action_space.n
            self.is_continuous = False
        else:
            #self.actions = self.env.action_space.shape[0]
            self.is_continuous = True

        self.actions = 1

        self.create_network()
        
        self.tf_sess.run(tf.initialize_all_variables())
                
        # Replay buffer
        self.replay = []

    def create_network(self):
        networks = {}

        with tf.variable_scope('q_net'):

            # Input parameters
            x = networks['x'] = tf.placeholder(tf.float32, shape=[None, self.states], \
                            name='states')
            u = networks['u'] = tf.placeholder(tf.float32, shape=[None, self.actions], \
                            name='actions')

            # hidden layers
            hidden_nodes = 100
            init = 1./hidden_nodes/2

            hid0 = fully_connected(tf.concat(1, [x, u]), 10, \
                weights_initializer=tf.random_normal_initializer(init, 0.01), \
                biases_initializer=tf.random_normal_initializer(init, 0.01), \
                activation_fn=tf.tanh)
            hid1 = fully_connected(hid0, 20, \
                weights_initializer=tf.random_normal_initializer(init, 0.01), \
                biases_initializer=tf.random_normal_initializer(init, 0.01), \
                activation_fn=tf.tanh)
            hid2 = fully_connected(hid1, 10, \
                weights_initializer=tf.random_normal_initializer(init, 0.01), \
                biases_initializer=tf.random_normal_initializer(init, 0.01), \
                activation_fn=tf.tanh)

            # Output parameters
            Q = networks['Q'] = fully_connected(hid2, 1, \
                weights_initializer=tf.random_normal_initializer(0.5, 0.1), \
                biases_initializer=tf.random_normal_initializer(0.5, 0.1))

            # Describe loss functions.
            y_ = networks['y_'] = tf.placeholder(tf.float32, [None, 1], name='y_i')
            networks['loss'] = tf.reduce_mean(tf.squared_difference(y_, \
                            Q), name='loss')
                            
            networks['optimize'] = tf.train.AdamOptimizer(learning_rate=self.alpha) \
                        .minimize(networks['loss'])
        
        self.tensors = networks

        return
    
    def reset(self):
        self.epsilon *= self.epsilon_decay
        return

    def get_action(self, state):
        # Get values for all actions.
        action0 = self.tf_sess.run(self.tensors['Q'], \
                        feed_dict={
                            self.tensors['x']: [state], 
                            self.tensors['u']: [[0]]
                            })
        action1 = self.tf_sess.run(self.tensors['Q'], \
                        feed_dict={
                            self.tensors['x']: [state], 
                            self.tensors['u']: [[1]]
                            })
        action = [action0[0][0], action1[0][0]]

        action_sm = softmax(action)
        #print action, action_sm
        
        action_sm += np.random.normal(0, self.epsilon, 2)
        
        return np.argmax(action_sm)
    
    def update(self, state, action, reward, state_prime, done):
        #reward = reward ** 1.5
        if done:
            reward = -reward

        #print state, action, reward, state_prime
        reward_p = reward + self.gamma * self.tf_sess.run(self.tensors['Q'], \
                feed_dict={
                    self.tensors['x']: [state_prime],
                    self.tensors['u']: [[self.get_action(state_prime)]]
                })[0][0]

        self.replay.append((state, action, reward, reward_p))

        m = self.update_samples

        for _ in range(self.update_steps):
            # Get m samples from self.replay
            if m > len(self.replay):
                m = len(self.replay)
            replays = random.sample(self.replay, m)
            x = [state]
            u = [[action]]
            y_ = [[reward_p]]
            for transition in replays:
                x.append(transition[0])
                u.append([transition[1]])
                y_.append([transition[3]])
            
            # minimize loss for y_i, x, u.
            self.tf_sess.run(self.tensors['optimize'], \
                feed_dict={
                    self.tensors['x']: x,
                    self.tensors['u']: u,
                    self.tensors['y_']: y_
                })

        return
