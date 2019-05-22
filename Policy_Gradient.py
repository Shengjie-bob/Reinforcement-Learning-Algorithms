#coding :utf-8
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gym

#gym 环境配置
env = gym.make('CartPole-v0')

#超参数的设置
D = 4 #输入
H = 10 #隐藏
batch_size = 5 #一次输入五次游戏
learning_rate  = 1e-2
gamma = 0.99 #奖励的折扣率

#定义网络结构
observations = tf.placeholder(tf.float32,shape=[None,D],name='input_data')
W1 = tf.get_variable(name='W1',shape=[D,H],dtype=tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
B1 = tf.get_variable(name='B1',shape=[H],initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,W1)+B1)

W2 = tf.get_variable(name='W2',shape=[H,1],dtype=tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
B2 = tf.get_variable(name='B2',shape=[1],initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)+B2
probability = tf.nn.sigmoid(score)

#定义训练中的loss相关的变量
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1],name='input_action')
advantages = tf.placeholder(tf.float32,name='Ai')  #reward_signal

#定义loss值
loglik = tf.log(input_y*(input_y-probability)+(1-input_y)*(input_y+probability))
loss = -tf.reduce_mean(loglik*advantages)
newGrads = tf.gradients(loss,tvars)
#定义一种简化的训练
# train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

#梯度的计算  此方法使得梯度可见 可以看到模型中梯度的变化
adam = tf.train.AdamOptimizer(learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

#计算折扣率之后的期望奖励
def discount_rewards(r):
    discount_r = np.zeros_like(r)
    running_add =0
    for t in reversed(range(0,r.size)):
        running_add = running_add *gamma +r[t]
        discount_r[t] =running_add
    return discount_r

#初始的参数设置
xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()

with tf.Session() as sess:
    rendering = False
    sess.run(init)

    observation = env.reset()
    # gradBuffer会存储梯度，此处做一初始化
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    while episode_number<=total_episodes:

        if reward_sum/batch_size >180 or rendering == True:
            env.render()
            rendering = True

        x= np.reshape(observation,[1,D])

        tfprob = sess.run(probability,feed_dict={observations:x})

        #输出概率下的action
        action = 1 if np.random.uniform() < tfprob else 0


        #记录每一步的观测和输出
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        #执行action
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward)

        if done:
            episode_number +=1
            # 将xs、ys、drs从list变成numpy数组形式
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []  # reset array memory

            #对reward做期望奖励  需要输入其中才可以做训练
            discount_epr = discount_rewards(epr)

            discount_epr -= np.mean(discount_epr)
            discount_epr//=np.std(discount_epr)

            tGrad= sess.run(newGrads,feed_dict={observations:epx,input_y:epy,advantages:discount_epr})

            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # 每batch_size局游戏，就将gradBuffer中的梯度真正更新到policy网络中
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # 打印一些信息
                print('Episode: %d ~ %d Average reward: %f.  ' % (
                episode_number - batch_size + 1, episode_number, reward_sum // batch_size))

                # 当我们在batch_size游戏中平均能拿到200的奖励，就停止训练
                if reward_sum // batch_size >= 200:
                    print("Task solved in", episode_number, 'episodes!')
                    break

                reward_sum = 0

            observation = env.reset()

print(episode_number, 'Episodes completed.')