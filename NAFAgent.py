import random
import gym
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.framework import get_variables

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

class NAFAgent:
    def __init__(self, env):
        # Initialive discounts, networks, EVERYTHING!
        self.gamma = 0.95
        self.tau = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = 0.9

        self.update_samples = 100
        self.update_steps = 10
        self.env = env
        self.tf_sess = tf.InteractiveSession()
        
        # Get state and action counts.
        self.states = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.actions = self.env.action_space.n
        else:
            self.actions = self.env.action_space.shape[0]

        self.network = {}
        for name in ['nqn', 'nqn_p']:
            self.create_network(name)
        
        self.tf_sess.run(tf.initialize_all_variables())
        
        # Buld update network:
        self.update_vars = {}
        for nqn_var, nqn_p_var in zip(self.network['nqn']['vars'], \
                self.network['nqn_p']['vars']):
            # Set initial values.
            self.tf_sess.run(nqn_p_var.assign(nqn_var))

            # Create tau update functions
            self.update_vars[nqn_p_var.name] = nqn_p_var.assign\
                    (self.tau * nqn_var + (1. - self.tau) * nqn_p_var)

        
        # Replay buffer
        self.replay = []

    def create_network(self, name):
        networks = {}

        with tf.variable_scope(name):

            # Input parameters
            networks['x'] = tf.placeholder(tf.float32, shape=[None, self.states], \
                            name='states')
            networks['u'] = tf.placeholder(tf.float32, shape=[None, self.actions], \
                            name='actions')

            # hidden layers
            hidden_nodes = 50
            hid0 = fully_connected(networks['x'], hidden_nodes, \
                weights_initializer=tf.random_normal_initializer(0.01, 0.001), \
                biases_initializer=tf.random_normal_initializer(0.01, 0.001), \
                activation_fn=tf.tanh)
            hid1 = fully_connected(hid0, hidden_nodes, \
                weights_initializer=tf.random_normal_initializer(0.01, 0.001), \
                biases_initializer=tf.random_normal_initializer(0.01, 0.001), \
                activation_fn=tf.tanh)

            # Output parameters
            networks['V'] = fully_connected(hid1, 1, \
                weights_initializer=tf.random_normal_initializer(0.01, 0.001), \
                biases_initializer=tf.random_normal_initializer(0.01, 0.001))
            networks['mu'] = fully_connected(hid1, self.actions, \
                weights_initializer=tf.random_normal_initializer(0.01, 0.001), \
                biases_initializer=tf.random_normal_initializer(0.01, 0.001))
            l = fully_connected(hid1, (self.actions * (self.actions + 1))/2,
                weights_initializer=tf.random_normal_initializer(0.02, 0.001), \
                biases_initializer=tf.random_normal_initializer(0.02, 0.001))
            
            # Build A(x, u)
            axis_T = 0
            rows = []

            # Identify diagonal 
            for i in xrange(self.actions):
                count = self.actions - i

                # Create a row with the diagonal elements exponentiated.
                diag = tf.exp(tf.slice(l, (0, axis_T), (-1, 1)))
                # Create the "other" elements of the row.
                others = tf.slice(l, (0, axis_T + 1), (-1, count - 1))

                # Assemble them into a full row.
                row = tf.pad(tf.concat(1, (diag, others)), \
                                ((0, 0), (i, 0)))

                # Add each row to a list for L(x)
                rows.append(row)

                axis_T += count

            # Assemble L(x) and matmul by its transpose.
            networks['L'] = tf.transpose(tf.pack(rows, axis=1), (0, 2, 1))
            networks['P'] = P = tf.batch_matmul(networks['L'], tf.transpose(networks['L'], (0, 2, 1)))

            mu_u = tf.expand_dims(networks['u'] - networks['mu'], -1)

            # Combine the terms
            A = (-1./2.) * tf.batch_matmul(tf.transpose(mu_u, [0, 2, 1]), \
                            tf.batch_matmul(P, mu_u))

            # Finally convert it back to a useable tensor...
            networks['A'] = tf.reshape(A, [-1, 1])

            networks['Q'] = networks['A'] + networks['V']

            # Describe loss functions.
            networks['y_'] = tf.placeholder(tf.float32, [None, 1], name='y_i')
            networks['loss'] = tf.reduce_mean(tf.squared_difference(networks['y_'], \
                            tf.squeeze(networks['Q'])), name='loss')

            # GradientDescent
            networks['gdo'] = tf.train.GradientDescentOptimizer(learning_rate=0.01) \
                        .minimize(networks['loss'])
        
        self.network[name] = networks
        self.network[name]['vars'] = get_variables(name)

        return

    def update_target(self):
        for variable in self.network['nqn_p']['vars']:
            self.tf_sess.run(self.update_vars[variable.name])
        return
    
    def reset(self):
        self.epsilon *= self.epsilon_decay
        return

    def get_action(self, state):
        # Get values for all actions.
        action = self.tf_sess.run(self.network['nqn']['mu'], \
                        feed_dict={self.network['nqn']['x']: [state]})
        
        print state, action
        # Softmax and add noise.
        softly = softmax(action[0] + np.random.normal(0, self.epsilon, self.actions))

        # Pick the best.
        action = np.argmax(softly)
        #print action

        return action
    
    def update(self, state, action, reward, state_prime, done):
        if done:
            reward = 0.

        self.replay.append((state, action, reward, state_prime))
        #print action, reward, done
        m = self.update_samples

        for _ in range(self.update_steps):
            # Get m samples from self.replay
            if m > len(self.replay):
                m = len(self.replay)
            replays = random.sample(self.replay, m)
            x = []
            u = []
            x_p = []
            y_ = []
            for replay in replays:
                x.append(replay[0])

                x_p.append(replay[3])

                # one-hot encode action.
                u_tmp = [0] * self.actions
                u_tmp[replay[1]] = 1.
                u.append(u_tmp)

                # self.nqn_y_ fed from r + self.gamma * V'(s_p)
                y_.append(replay[2])
            
            V_p = self.tf_sess.run(self.network['nqn_p']['V'], \
                    feed_dict={self.network['nqn_p']['x']: x_p})
            
            y_ = [[temp1 + temp2[0]] for temp1, temp2 in zip(y_, self.gamma * V_p)]
            
            #print 'L={0}'.format(self.tf_sess.run(self.network['nqn']['L'], feed_dict={self.network['nqn']['x']: x, self.network['nqn']['u']: u}))
            
            #print 'u: {0}, y_:{1}'.format(u, y_)
            # minimize nqn_L for y_i, x, u.
            self.tf_sess.run(self.network['nqn']['gdo'], \
                feed_dict={
                    self.network['nqn']['x']: x,
                    self.network['nqn']['u']: u,
                    self.network['nqn']['y_']: y_
                })

            # update target network. 
            self.update_target()

        return