import random
import gym
import numpy as np
import tensorflow as tf

class NAFAgent:
    def __init__(self, env):
        self.env = env
        self.tf_sess = tf.InteractiveSession()
        
        # Get input and output counts.
        self.inputs = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete.Discrete):
            self.outputs = self.env.action_space.n
        else:
            self.outputs = self.env.action_space.shape[0]
        
        # Input layer
        self.nqn_x = tf.placeholder(tf.float32, shape=[None, \
                        self.inputs])
        # Output layer
        self.nqn_y_ = tf.placeholder(tf.float32, shape=[None, \
                        self.outputs])
        
        # Weights and biases
        self.W = tf.Variable(tf.zeros([self.inputs, self.outputs]))
        self.b = tf.Variable(tf.zeros([self.outputs]))

        self.tf_sess.run(tf.initialize_all_variables())

        self.nqn_y = tf.nn.softmax(tf.matmul(self.nqn_x, self.W) + self.b)

        # Minimizing mean squared error
        self.sum_sq_err = tf.reduce_mean(tf.pow(tf.sub(self.nqn_y_, self.nqn_y), 2.0))

        # Define training step.
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.sum_sq_err)

        # Replay buffer
        self.replay = []
    
    def reset(self):
        return

    def get_action(self, state):
        return self.env.action_space.sample()
    
    def update(self, state, action, reward, state_prime, done):
        self.replay.append((state, action, reward, state_prime))
        for i in range(50):
            # Get m samples from self.replay
            # self.nqn_y_ fed from r + self.gamma * V'(s_p)
            self.train_step.run()
            # update theta Q' 
        return