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
        self.tau = 0.05
        self.epsilon = 1.0
        self.epsilon_decay = 0.98

        self.alpha = 0.0001

        self.update_samples = 100
        self.update_steps = 10

        self.hidden_layers = 5
        self.hidden_nodes = 20

        print self.gamma, self.tau, self.epsilon, self.epsilon_decay, \
            self.alpha, self.update_samples, self.update_steps, self.hidden_layers, \
            self.hidden_nodes

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

        self.one_hot = self.tf_sess.run(tf.one_hot(range(2), 2))
        
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
            init = 1./self.hidden_nodes/self.actions

            hid = networks['x']
            hid = fully_connected(hid, self.hidden_nodes, \
                    weights_initializer=tf.random_normal_initializer(init, init/5), \
                    biases_initializer=tf.random_normal_initializer(init, init/5), \
                    activation_fn=tf.tanh)
            for i in xrange(self.hidden_layers-1):
                hid = fully_connected(hid, self.hidden_nodes, \
                    weights_initializer=tf.random_normal_initializer(init, init/5), \
                    biases_initializer=tf.random_normal_initializer(init, init/5), \
                    activation_fn=tf.nn.relu)
                
                if i + 1 % 3 == 0:
                    # softmax every third layer
                    hid = tf.nn.softmax(hid)
            
            #hid = tf.nn.softmax(hid)

            # Output parameters
            networks['V'] = fully_connected(hid, self.actions, \
                weights_initializer=tf.random_normal_initializer(1., 0.1), \
                biases_initializer=tf.random_normal_initializer(0., 0.1))
            networks['mu'] = fully_connected(hid, self.actions, \
                weights_initializer=tf.random_normal_initializer(1., 0.1), \
                biases_initializer=tf.random_normal_initializer(0., 0.1))
            networks['mu_out'] = tf.nn.softmax(networks['mu'])

            # Linear output layer
            l = fully_connected(hid, (self.actions * (self.actions + 1))/2,
                weights_initializer=tf.random_normal_initializer(1., 0.1), \
                biases_initializer=tf.random_normal_initializer(0., 0.1))
            
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
            networks['P'] = P = tf.batch_matmul(networks['L'], \
                tf.transpose(networks['L'], (0, 2, 1)))

            mu_u = tf.expand_dims(networks['u'] - networks['mu'], -1)

            # Combine the terms
            p_mu_u = tf.batch_matmul(P, mu_u, name='Pxmu_u')
            p_mess = tf.batch_matmul(tf.transpose(mu_u, [0, 2, 1]), p_mu_u, name='mu_u_TxPxmu_u')
            networks['A'] = tf.mul(-1./2., p_mess, name='A')

            networks['Q'] = tf.add(networks['A'], networks['V'], name='Q_func')

            # Describe loss functions.
            networks['y_'] = tf.placeholder(tf.float32, [None, 1], name='y_i')
            networks['loss'] = tf.reduce_mean(tf.squared_difference(networks['y_'], \
                            tf.squeeze(networks['Q'])), name='loss')

            # GradientDescent
            networks['gdo'] = tf.train.AdamOptimizer(learning_rate=self.alpha, epsilon=0.5) \
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

    def get_action(self, state, report=False):
        # Get values for all actions.
        #action = []
        #for act in xrange(self.actions):
        #    temp = self.tf_sess.run(self.network['nqn']['Q'], \
        #                    feed_dict={
        #                        self.network['nqn']['x']: [state], 
        #                        self.network['nqn']['u']: [self.one_hot[act]]
        #                    })
            #print temp
            #if temp[0][0] <= 0.0:
        #    report = True
        #    action.append(temp[0][0])
        
        
        #action_sm = softmax(action)

        #if report:
            #print 'actions: {0}'.format(action)
            #print 'softmax: {0}'.format(action_sm)
            #print 'mu: {0}'.format(self.tf_sess.run(self.network['nqn']['mu'],\
            #     feed_dict={self.network['nqn']['x']: [state]}))

        action = self.tf_sess.run(self.network['nqn']['mu'],\
                 feed_dict={self.network['nqn']['x']: [state]})
        action_sm = self.tf_sess.run(self.network['nqn']['mu_out'],\
                 feed_dict={self.network['nqn']['x']: [state]})

        if report:
            print 'mu: {0}'.format(action)
            print 'softmax: {0}'.format(action_sm)
        
        action_sm += np.random.normal(0, self.epsilon, self.actions)
        
        return np.argmax(action_sm)
    
    def update(self, state, action, reward, state_prime, done):
        if done:
            reward = -reward

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

                u.append(self.one_hot[replay[1]])

                # self.nqn_y_ fed from r + self.gamma * V'(s_p)
                y_.append(replay[2])
            
            V_p = self.tf_sess.run(self.network['nqn_p']['V'], \
                    feed_dict={self.network['nqn_p']['x']: x_p})

            #print y_, V_p

            y_ = [[temp1 + temp2[0]] for temp1, temp2 in zip(y_, self.gamma * V_p)]
            #print 'y_i: {0}'.format(y_)
            
            #print 'u: {0}, y_:{1}'.format(u, y_)
            # minimize loss function for y_i, x, u.
            self.tf_sess.run(self.network['nqn']['gdo'], \
                feed_dict={
                    self.network['nqn']['x']: x,
                    self.network['nqn']['u']: u,
                    self.network['nqn']['y_']: y_
                })

            # update target network. 
            self.update_target()

        return