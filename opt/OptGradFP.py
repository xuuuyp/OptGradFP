import numpy as np
import gym
import matplotlib.pyplot as plt
import random
RENDER = False
import tensorflow as tf

'''
连续动作
使用的是高斯策略，用神经网络参数化高斯分布的均值和方差
状态空间为 2
动作空间为 1   其取值范围在[-2,2]之间
注意其误差的构建
'''


# 定义策略网络
class Policy_Net:
    def __init__(self, args, agent_id, env):
        self.agent_id = agent_id
        self.args = args
        self.action_bound = [[-1.], [1.0]]
        self.learning_rate = 0.0001
        #   输入特征的维数
        self.n_features = env.observation_space[agent_id].shape[0]
        #   输出动作空间的维数
        self.n_actions = 5  # 注意动作维度只有一个
        #   1.1 输入层
        self.obs = tf.placeholder(tf.float32, shape=[None, self.n_features])
        #   1.2.第一层隐含层
        self.f1 = tf.layers.dense(inputs=self.obs, units=200, activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                  bias_initializer=tf.constant_initializer(0.1))
        #   1.3 第二层，均值，需要注意的是激活函数为tanh，使得输出在-1~+1
        mu = tf.layers.dense(inputs=self.f1, units=self.n_actions, activation=tf.nn.tanh,
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                             bias_initializer=tf.constant_initializer(0.1))
        #   1.3 第二层，标准差
        sigma = tf.layers.dense(inputs=self.f1, units=self.n_actions, activation=tf.nn.softplus,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                bias_initializer=tf.constant_initializer(0.1))
        self.mu = mu
        self.sigma = sigma
        # 定义带参数的正态分布
        self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        #   根据正态分布采样一个动作
        self.action = tf.clip_by_value(self.normal_dist.sample(1), self.action_bound[0], self.action_bound[1])
        #   1.5 当前动作
        self.current_act = tf.placeholder(tf.float32, [None, 5])
        self.current_reward = tf.placeholder(tf.float32, [None, 5])
        #   TODO 2. 构建损失函数
        log_prob = self.normal_dist.log_prob(self.current_act)
        self.loss = tf.reduce_mean(log_prob * self.current_reward + 0.01 * self.normal_dist.entropy())

        #   3. 定义一个优化器
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.loss)
        #   4. tf工程
        self.sess = tf.Session()
        #   5. 初始化图中的变量
        self.sess.run(tf.global_variables_initializer())
        #   6.定义保存和恢复模型
        self.saver = tf.train.Saver()

    #   依概率选择动作
    def choose_action(self, state, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            u = self.sess.run(self.action, {self.obs: state})
            u = u[0][0]
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()

    #   定义训练
    def train_step(self, state_batch, label_batch, reward_batch):
        state_batch = random.sample(state_batch, 5)
        label_batch = random.sample(label_batch, 5)
        reward_batch = random.sample(reward_batch, 5)
        loss, _ = self.sess.run([self.loss, self.train_op],
                                    feed_dict={self.obs: state_batch, self.current_act: label_batch,
                                               self.current_reward: reward_batch})
        return loss

    #   定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)

    #   定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)





