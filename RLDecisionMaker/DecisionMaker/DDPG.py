import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from functionality import *

#####################  hyper parameters  ####################
lr_actor = 0.0001
lr_critic = 0.0002
gamma = 0.9
tau = 0.01
batch_size = 32
is_output_graph = False

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, state_dim, resource_dim, offload_dim):
        tf.reset_default_graph()
        self.memory_capacity = 300
        
        # dimension
        self.state_dim = state_dim # 3 + 1 + 3 + 1
        self.action_dim = resource_dim + offload_dim # 2 + 3
        self.resource_dim = resource_dim # 2
        self.offload_dim = offload_dim # 3
        self.filename = "train_data.json"

        # # bound
        # self.resource_bound = resource_bound

        # input placeholder
        self.S = tf.placeholder(tf.float32, [None, state_dim], 'S')
        self.S_ = tf.placeholder(tf.float32, [None, state_dim], 'S_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        # memory
        self.memory, self.memory_count = init_memory(self.filename, self.memory_capacity, self.state_dim, self.action_dim)  # state_dim + action_dim + r_dim + state_dim
        print("memory_count = ", self.memory_count)

        # session
        self.sess = tf.Session()

        # input & output definition
        self.a = self.build_actor(self.S, )
        q_predict = self.build_critic(self.S, self.a)

        # Obtain Actor and Critic parameters
        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        # soft replacement
        ema = tf.train.ExponentialMovingAverage(decay=1 - tau)

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        # Update the params of the target Network
        target_update = [ema.apply(actor_params), ema.apply(critic_params)]

        a_ = self.build_actor(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_predict_ = self.build_critic(self.S_, a_, reuse=True, custom_getter=ema_getter)

        # Actor learn()
        actor_loss = -tf.reduce_mean(q_predict)
        self.actor_optimizer = tf.train.AdamOptimizer(lr_actor).minimize(actor_loss, var_list=actor_params)

        # Critic learn()
        # We first update the params of the target Network
        # Then update the params of the Critic Network
        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + gamma * q_predict_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q_predict)
            self.critic_optimizer = tf.train.AdamOptimizer(lr_critic).minimize(td_error, var_list=critic_params)

        # Initialize the variables before beginning training
        self.sess.run(tf.global_variables_initializer())

        if is_output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(self.memory_capacity, size=batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.state_dim]
        ba = bt[:, self.state_dim: self.state_dim + self.action_dim]
        br = bt[:, -self.state_dim - 1: -self.state_dim]
        bs_ = bt[:, -self.state_dim:]

        self.sess.run(self.actor_optimizer, {self.S: bs})
        self.sess.run(self.critic_optimizer, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        print()
        print("Training is going on!")
        print()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_count % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.memory_count += 1

    def build_actor(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            n_l = 50
            net = tf.layers.dense(s, n_l, activation=tf.nn.relu, name='l1', trainable=trainable)

            # resource ( 0 - r_bound)
            layer_r0 = tf.layers.dense(net, n_l, activation=tf.nn.relu, name='r_0', trainable=trainable)
            layer_r1 = tf.layers.dense(layer_r0, n_l, activation=tf.nn.relu, name='r_1', trainable=trainable)
            layer_r2 = tf.layers.dense(layer_r1, n_l, activation=tf.nn.relu, name='r_2', trainable=trainable)
            layer_r3 = tf.layers.dense(layer_r2, n_l, activation=tf.nn.relu, name='r_3', trainable=trainable)
            layer_r4 = tf.layers.dense(layer_r3, self.resource_dim, activation=tf.nn.relu, name='r_4',
                                       trainable=trainable)

            # offloading (probability: 0 - 1)
            # layer
            layer = ["layer0" + str(layer) for layer in range(4)]
            # name
            name = ["name0" + str(layer) for layer in range(4)]
            # user
            user = "user0"
            # softmax
            softmax = "softmax0"
            layer[0] = tf.layers.dense(net, n_l, activation=tf.nn.relu, name=name[0], trainable=trainable)
            layer[1] = tf.layers.dense(layer[0], n_l, activation=tf.nn.relu, name=name[1], trainable=trainable)
            layer[2] = tf.layers.dense(layer[1], n_l, activation=tf.nn.relu, name=name[2], trainable=trainable)
            layer[3] = tf.layers.dense(layer[2], self.offload_dim, activation=tf.nn.relu, name=name[3], trainable=trainable)
            user = tf.nn.softmax(layer[3], name=softmax)

            # concate
            a = tf.concat([layer_r4, user], 1) # 2 + 3
            return a

    def build_critic(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        # Q value (0 - inf)
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l = 50
            w1_s = tf.get_variable('w1_s', [self.state_dim, n_l], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.action_dim, n_l], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l], trainable=trainable)
            net_1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_2 = tf.layers.dense(net_1, n_l, activation=tf.nn.relu, trainable=trainable)
            net_3 = tf.layers.dense(net_2, n_l, activation=tf.nn.relu, trainable=trainable)
            net_4 = tf.layers.dense(net_3, n_l, activation=tf.nn.relu, trainable=trainable)
            return tf.layers.dense(net_4, 1, activation=tf.nn.relu, trainable=trainable)  # Q(s,a)












