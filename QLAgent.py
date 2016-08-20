import random

import numpy as np

import gym
import tensorflow as tf
from tensorflow.contrib.framework import get_variables
from tensorflow.contrib.layers import fully_connected

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

class QLAgent:
    def __init__(self, env):
        # Initialive discounts, networks, EVERYTHING!
        self.gamma = 0.6
        self.epsilon = 1.0
        self.epsilon_decay = 0.95

        self.alpha = 0.0005

        self.hidden_layers = 3
        self.hidden_nodes = 100

        self.update_samples = 100
        self.update_steps = 10
        self.env = env
        self.tf_sess = tf.Session()
        
        # Get state and action counts.
        self.states = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.actions = self.env.action_space.n
            self.is_continuous = False
        else:
            self.actions = self.env.action_space.shape[0]
            self.is_continuous = True

        self.one_hot = self.tf_sess.run(tf.one_hot(range(self.actions), self.actions))

        self.create_network()
        
        self.tf_sess.run(tf.initialize_all_variables())
                
        # Replay buffer
        self.replay = []

    def create_network(self):
        networks = {}

        with tf.variable_scope('q_net'):

            # Input parameters
            x = networks['x'] = tf.placeholder(tf.float32, \
                            shape=[None, self.states], name='states')
            u = networks['u'] = tf.placeholder(tf.float32, \
                            shape=[None, self.actions], name='actions')

            # hidden layers
            init = 1./self.hidden_nodes/self.actions

            hid = tf.concat(1, [x,  u])
            hid = fully_connected(hid, self.hidden_nodes, \
                weights_initializer=tf.random_normal_initializer(init, init/5), \
                biases_initializer=tf.random_normal_initializer(init, init/5), \
                activation_fn=tf.tanh)

            for i in xrange(self.hidden_layers-1):
                hid = fully_connected(hid, self.hidden_nodes, \
                    weights_initializer=tf.random_normal_initializer(init, init/5), \
                    biases_initializer=tf.random_normal_initializer(init, init/5), \
                    activation_fn=tf.nn.relu)

            # Output parameters
            pos_layer = fully_connected(hid, 1, \
                weights_initializer=tf.random_normal_initializer(1./self.actions, 0.1), \
                biases_initializer=tf.random_normal_initializer(1./self.actions, 0.1))
            neg_layer = tf.neg(fully_connected(hid, 1, \
                weights_initializer=tf.random_normal_initializer(1./self.actions, 0.1), \
                biases_initializer=tf.random_normal_initializer(1./self.actions, 0.1)))

            Q = networks['Q'] = pos_layer + neg_layer

            # Describe loss functions.
            y_ = networks['y_'] = tf.placeholder(tf.float32, [None, 1], name='y_i')


            # Tensor outputs to calculate y_i values
            networks['reward'] = tf.placeholder(tf.float32, [None, 1], name='reward')
            networks['y_calc'] = tf.add(networks['reward'], tf.mul(Q, self.gamma))

            networks['loss'] = tf.reduce_mean(tf.squared_difference(y_, \
                            Q), name='loss')
                            
            networks['optimize'] = tf.train.AdamOptimizer(\
                        learning_rate=self.alpha) \
                        .minimize(networks['loss'])
        
        self.tensors = networks
        return
    
    def reset(self):
        self.epsilon *= self.epsilon_decay
        return

    def get_action(self, state, report=False):
        # Get values for all actions.
        action = []
        for act in xrange(self.actions):
            temp = self.tf_sess.run(self.tensors['Q'], \
                            feed_dict={
                                self.tensors['x']: [state], 
                                self.tensors['u']: [self.one_hot[act]]
                            })
            action.append(temp[0][0])
        
        if report:
            print 'Actions: {0}'.format(action)

        action_sm = softmax(action)
        action_sm += np.random.normal(0, self.epsilon, self.actions)
        
        return np.argmax(action_sm)

    def update(self, state, action, reward, state_prime, done):
        #reward = reward ** 1.5
        if done:
            reward = -reward

        #print state, action, reward, state_prime
        action = self.one_hot[action]
        next_act = self.one_hot[self.get_action(state_prime)]
        reward_p = reward + self.gamma * self.tf_sess.run(self.tensors['Q'], \
                feed_dict={
                    self.tensors['x']: [state_prime],
                    self.tensors['u']: [next_act]
                })[0][0]

        #print state, action, reward, reward_p

        m = self.update_samples

        for _ in range(self.update_steps):
            # Get m samples from self.replay
            if m > len(self.replay):
                m = len(self.replay)
            replays = random.sample(self.replay, m)
            
            # Include the current transition so it is involved at least once.
            x = [state]
            u = [action]
            y_ = [[reward_p]]

            nx = [state_prime]
            nu = [next_act]
            nr = [[reward]]

            for transition in replays:
                #print transition
                x.append(transition[0])
                u.append(transition[1])

                nx.append(transition[3])
                nu.append(self.one_hot[self.get_action(transition[3])])
                nr.append([transition[2]])
                #y_.append([discount_r])

            #print nr, nx
            y_ = self.tf_sess.run(self.tensors['y_calc'], \
                    feed_dict={
                        self.tensors['x']: nx,
                        self.tensors['u']: nu,
                        self.tensors['reward']: nr
                    })
                
            #print y_

            # minimize loss for y_i, x, u.
            self.tf_sess.run(self.tensors['optimize'], \
                feed_dict={
                    self.tensors['x']: x,
                    self.tensors['u']: u,
                    self.tensors['y_']: y_
                })
        

        self.replay.append((state, action, reward, state_prime))
        return
