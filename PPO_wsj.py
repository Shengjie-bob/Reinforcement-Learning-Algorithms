import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LA = 0.0001
C_LA = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
METHOD = [dict(name='kl_pen', kl_target=0.01, lam=0.5),
          dict(name='clip', epsilon=0.2)][1]


class PPO(object):
    def __init__(self):

        self.sess = tf.Session()

        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], name='state')
        self.tfr = tf.placeholder(tf.float32, [None, 1], name='reward')
        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], name='action')
        self.tfs_ = tf.placeholder(
            tf.float32, [
                None, S_DIM], name='next_state')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], name='advantage')
        self.tfv = tf.placeholder(
            tf.float32, [
                None, 1], name='next_state')
        with tf.variable_scope('Cristic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.advantage = (
                self.tfr +self.tfv) - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain = tf.train.AdamOptimizer(C_LA).minimize(self.closs)

        # with tf.variable_scope('Actor'):
        pi, pi_params = self._build_anet('a_new', True)
        pi_old, pi_old_params = self._build_anet('a_old', False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.variable_scope('oldpi_update'):
            self.oldpi_op = [
                oldp.assign(p) for p, oldp in zip(
                    pi_params, pi_old_params)]  # 这是十分好的一种给数的方式
        with tf.variable_scope('a_loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / pi_old.prob(self.tfa)       #新旧的策略的更新在于对不同概率分布下该动作的概率所以需要输入动作变量
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lamda')
                kl = tf.distributions.kl_divergence(pi_old, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:
                self.aloss = -(tf.reduce_mean(tf.minimum(surr,
                                                        tf.clip_by_value(ratio,
                                                                         1. - METHOD['epsilon'],
                                                                         1. + METHOD['epsilon']) * self.tfadv)))

        with tf.variable_scope('a_train'):
            self.atrain_op = tf.train.AdamOptimizer(A_LA).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r, v):
        self.sess.run(self.oldpi_op)
        adv = self.sess.run(
            self.advantage,
            feed_dict={
                self.tfr: r,
                self.tfs: s,
                self.tfv: v})

        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run([self.atrain_op, self.kl_mean], feed_dict={
                                      self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4 * METHOD['kl_target']:
                    break
            if kl < METHOD['kl_target'] / 1.5:
                METHOD['lam'] /= 2
            if kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)
        else:
            [self.sess.run(self.atrain_op,
                           {self.tfs: s,
                            self.tfa: a,
                            self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        [self.sess.run(self.ctrain,
                       {self.tfs: s,
                        self.tfr: r,
                        self.tfv: v}) for _ in range(C_UPDATE_STEPS)]

    def choose_action(self, s):
        s = s[np.newaxis, :]  # 主要由于是batch输入可能会造成维度不对
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            normal_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=name)  # 可以做到将整个网络的参数得到
        return normal_dist, params

    def get_v_(self, s_):
        s_ = s_[np.newaxis,:]
        return self.sess.run(self.v, feed_dict={self.tfs: s_})


env = gym.make('Pendulum-v0').unwrapped

ppo = PPO()
RENDER=False
var = 3
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
    ep_r = 0
    for t in range(EP_LEN):
        if RENDER :
            env.render()
        a = ppo.choose_action(s)

        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append(r)
        v = ppo.get_v_(s_)
        buffer_v.append(v)
        s = s_
        ep_r += r
        if (t + 1) % BATCH == 0 or (t - 1) == EP_LEN:
            bs, ba, br, bv = np.vstack(buffer_s), np.vstack(
                buffer_a), np.vstack(buffer_r), np.vstack(buffer_v)

            buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
            ppo.update(bs, ba, br, bv)

    if ep == 0:
        all_ep_r.append(ep_r)
    else:
        all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
    if ep_r > -300:
        RENDER =True
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.show()
