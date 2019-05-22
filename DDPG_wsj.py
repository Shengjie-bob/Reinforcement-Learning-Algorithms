import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import time

np.random.seed(1)
tf.set_random_seed(1)

EP_MAX =200
EP_LEN =200
A_LA =0.001
C_LA =0.001
GAMMA =0.9
REPLACEMENT = [
    dict(name ='soft',tau =0.01),
    dict(name ='hard',rep_iter_a=600, rep_iter_c=500)
][1]

MEMORY_CAPACITY = 10000
BATCH_SIZE =32

RENDER =False
OUT_GRAPH = False

ENV_NAME = 'Pendulum-v0'



class Actor(object):
    def __init__(self,sess,action_dim,action_bound,learning_rate,replacement,S,S_):
        self.sess =sess
        self.a_dim =action_dim
        self.action_bound = action_bound
        self.lr =learning_rate
        self.replacement =replacement
        self.t_replace_counter = 0

        self.S,self.S_ =S,S_

        with tf.variable_scope('Actor'):
            self.a  = self._build_net(self.S,scope = 'eval_net',trainable =True)

            self.a_ = self._build_net(self.S_,scope ='target_net',trainable =False)

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/eval_net')
            self.t_params =tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter =0
            self.hard_replace = [tf.assign(t,e) for t ,e in zip(self.t_params,self.e_params)]

        else:
            self.soft_replace = [tf.assign(t,(1-self.replacement['tau'])*t+self.replacement['tau']*e)
                                 for t,e in zip(self.t_params,self.e_params)]

    def _build_net(self,s,scope,trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net  =tf.layers.dense(s,30,activation=tf.nn.relu,kernel_initializer=init_w,
                                  bias_initializer=init_b,name ='l1',trainable=trainable)

            with tf.variable_scope('a'):
                actions = tf.layers.dense(net,self.a_dim,activation=tf.nn.tanh,
                                          kernel_initializer=init_w,bias_initializer=init_b,
                                          name='a',trainable=trainable)
                scaled_a = tf.multiply(actions,self.action_bound,name='scaled_a')

            return scaled_a

    def learn(self,s):
        # self.sess.run(self.train_op,feed_dict={self.S:s})
        self.sess.run(self.atrain, feed_dict={self.S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self,s):
        s = s[np.newaxis,:]
        return self.sess.run(self.a, feed_dict={self.S: s})[0]  # single action

    def add_grads_to_graph(self,a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys =self.a,xs =self.e_params,grad_ys=a_grads)

        with tf.variable_scope('train_anet'):
            opt = tf.train.AdamOptimizer(-self.lr)   #由于要最大化函数值
            self.train_op =opt.apply_gradients(zip(self.policy_grads, self.e_params))

    def add_q_to_graph(self,q):
        with tf.variable_scope('a_train'):
            self.q = q
            self.a_loss = - tf.reduce_mean(self.q)
            self.atrain = tf.train.AdamOptimizer(self.lr).minimize(self.a_loss, var_list=self.e_params)

class Critic(object):
    def __init__(self,sess,state_dim,action_dim,learning_rate,gamma,replacement,S,R,S_,a,a_):
        self.sess =sess
        self.s_dim = state_dim
        self.a_dim =action_dim
        self.lr =learning_rate
        self.gamma =gamma
        self.replacement =replacement
        self.t_replace_counter =0

        self.S,self.R ,self.S_ = S, R, S_

        with tf.variable_scope('Critic'):
            # self.a = tf.stop_gradient(a)
            self.a = a

            self.q =self._build_net(self.S,self.a,scope ='eval_net',trainable =True)

            self.q_ =self._build_net(self.S_,a_,scope ='target_net',trainable =False)

            self.e_params =tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

            if self.replacement['name'] =='hard':
                self.hard_replace = [tf.assign(t,e) for t,e in zip(self.t_params,self.e_params)]
                self.t_replace_counter = 0
            else:
                self.soft_replace = [tf.assign(t,(1-self.replacement['tau'])*t+self.replacement['tau']*e)
                                     for t,e in zip(self.t_params,self.e_params)]

            with tf.variable_scope('target_q'):
                self.target_q = self.R + self.gamma*self.q_

            with tf.variable_scope('TD-error'):
                self.loss = tf.losses.mean_squared_error(labels=self.target_q, predictions=self.q)  #这个是关键

            with tf.variable_scope('C_train'):
                self.train_op =tf.train.AdamOptimizer(self.lr).minimize(self.loss,var_list=self.e_params)

            with tf.variable_scope('a_grads'):
                self.a_grads =tf.gradients(self.q,a)[0]

    def _build_net(self,s,a,scope,trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1= 30
                w1_s = tf.get_variable('w1_s',[self.s_dim,n_l1],initializer=init_w,trainable=trainable)
                w1_a =tf.get_variable('w1_a',[self.a_dim,n_l1],initializer=init_w,trainable=trainable)
                b1 = tf.get_variable('b1',[n_l1],initializer=init_b,trainable= trainable)
                net =tf.nn.relu(tf.matmul(s,w1_s)+tf.matmul(a,w1_a)+b1)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net,1,kernel_initializer=init_w,bias_initializer=init_b,trainable=trainable)

        return q

    def learn(self,s,a,r,s_):
        self.sess.run(self.train_op, feed_dict={self.S: s, self.a: a, self.R: r, self.S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]



class DDPG(object):
    def __init__(self,state_dim,action_dim,action_bound):
        self.sess =tf.Session()
        self.S =tf.placeholder(tf.float32,[None,state_dim],name='state')
        self.R =tf.placeholder(tf.float32,[None,1],name='reward')
        self.S_ =tf.placeholder(tf.float32,[None,state_dim],name='next_state')

        self.actor = Actor(self.sess,action_dim,action_bound,A_LA,REPLACEMENT,self.S,self.S_)
        self.critic =Critic(self.sess,
                              state_dim,
                              action_dim,
                              C_LA,GAMMA,
                              REPLACEMENT,
                              self.S,self.R,self.S_,
                              self.actor.a, self.actor.a_)
        # self.actor.add_grads_to_graph(self.critic.a_grads)
        self.actor.add_q_to_graph(self.critic.q)

        self.sess.run(tf.global_variables_initializer())
        self.M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)
        if OUT_GRAPH:
            tf.summary.FileWriter('logs/',self.sess.graph)

env = gym.make(ENV_NAME)
env = env.unwrapped

env.seed(1)

state_dim= env.observation_space.shape[0]
action_dim =env.action_space.shape[0]
action_bound =env.action_space.high

var =3
ddpg =DDPG(state_dim,action_dim,action_bound)

t1 =time.time()
all_ep_reward = []
for i in range(EP_MAX):
    s = env.reset()
    epr_reward =0
    for j in range(EP_LEN):
        if RENDER:
            env.render()

        a = ddpg.actor.choose_action(s)
        a = np.clip(np.random.normal(a,var),-2,2)
        s_,r,done,_  =env.step(a)

        ddpg.M.store_transition(s,a,r/10,s_)

        if ddpg.M.pointer > MEMORY_CAPACITY:
            var *=.9995
            b_M =ddpg.M.sample(BATCH_SIZE)
            b_s =b_M[:,:state_dim]
            b_a =b_M[:,state_dim: state_dim + action_dim]
            b_r = b_M[:, -state_dim - 1: -state_dim]
            b_s_ = b_M[:, -state_dim:]

            ddpg.critic.learn(b_s,b_a,b_r,b_s_)       #好好理解这是怎么传递的参数
            ddpg.actor.learn(b_s)


        s =s_
        epr_reward +=r

        if j == EP_LEN-1:
            print('Episode:', i, ' Reward: %i' % int(epr_reward), 'Explore: %.2f' % var, )
            if i ==0:
                all_ep_reward.append(epr_reward)
            else: all_ep_reward.append(all_ep_reward[-1]*0.9+0.1*epr_reward)
            if epr_reward > -300:
                RENDER = True
            break

print('Running time: ', time.time() - t1)

