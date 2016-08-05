import random
import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

class NAFAgent:
    def __init__(self, env):
        self.env = env
        self.tf_sess = tf.InteractiveSession()
        
        # Get input and output counts.
        # One extra output for V
        self.states = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.actions = self.env.action_space.n
        else:
            self.actions = self.env.action_space.shape[0]
        
        # Input parameters
        self.nqn_x = tf.placeholder(tf.float32, shape=[None, self.states], \
                        name='states')
        self.nqn_u = tf.placeholder(tf.float32, shape=[None, self.actions], \
                        name='actions')
                        
        # hidden layers
        hidden_nodes = 100
        hid0 = fully_connected(self.nqn_x, hidden_nodes)
        hid1 = fully_connected(hid0, hidden_nodes)

        # Output parameters
        self.nqn_V = fully_connected(hid1, 1, scope='V')
        self.nqn_mu = fully_connected(hid1, self.actions, scope='mu')
        l = fully_connected(hid1, self.actions, scope='L')
        
        # Build A(x, u)
        axis_T = 0
        rows = []

        # Identify diagonal 
        for i in xrange(self.actions):
            count = self.actions - i

            # Slice out diagonal rows for exponentiation.
            diag = tf.exp(tf.slice(l, (0, axis_T), (-1, 1)))
            others = tf.slice(l, (0, axis_T + 1), (-1, count - 1))
            row = tf.pad(tf.concat(1, (diag, others)), \
                            ((0, 0), (i, 0)))

            # Add each diagonal row to a list for L(x)
            rows.append(row)

            axis_T += count

        # Assemble L(x) and matmul by its transpose.
        self.nqn_L = tf.transpose(tf.pack(rows, axis=1), (0, 2, 1))
        P = tf.batch_matmul(self.nqn_L, tf.transpose(self.nqn_L, (0, 2, 1)))

        mu_u = tf.expand_dims(self.nqn_u - self.nqn_mu, -1)

        # Combine the terms
        A = (-1./2.) * tf.batch_matmul(tf.transpose(mu_u, [0, 2, 1]), \
                        tf.batch_matmul(P, mu_u))

        # Finally convert it back to a useable tensor...
        self.nqn_A = tf.reshape(A, [-1, 1])

        self.nqn_Q = self.nqn_A + self.nqn_V

        self.nqn_y_ = tf.placeholder(tf.float32, [None], name='y_i')
        self.nqn_loss = tf.reduce_mean(tf.squared_difference(self.nqn_y_, \
                        tf.squeeze(self.nqn_Q)), name='loss')

        # Replay buffer
        self.replay = []
    
    def reset(self):
        return

    def get_action(self, state):
        # Get values for all actions.
        action = self.tf_sess.run(self.nqn_mu, feed_dict={self.nqn_x: [state]})

        # Softmax and add noise.
        softly = softmax(action[0]) + np.random.normal(0, 0.1, self.actions)

        # Pick the best.
        action = np.argmax(softly)
        
        return action
    
    def update(self, state, action, reward, state_prime, done):
        state.append(action)
        self.replay.append((state, reward, state_prime))
        for i in range(50):
            # Get m samples from self.replay
            m = 50
            if m > len(self.replay):
                m = len(self.replay)
            replays = random.sample(self.replay, m)
            x = []
            y_ = []
            for replay in replays:
                x.append(replay[0])

                # self.nqn_y_ fed from r + self.gamma * V'(s_p)
                V_p = 0.5
                expected = [replay[1] + self.gamma * V_p]
                y_.append(expected)

            self.train_step.run(session=self.tf_sess, \
                        feed_dict={self.nqn_x: x, self.nqn_y_: y_})
            # update theta Q' 
        return