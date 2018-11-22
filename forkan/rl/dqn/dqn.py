import logging
import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug
from tabulate import tabulate

from forkan.utils import store_args
from forkan.schedules import LinearSchedule

from forkan.rl.losses import huber_loss
from forkan.rl.utils import vector_summary, scalar_summary, rename_latest_run, clean_dir
from forkan.rl.dqn.networks import build_network
from forkan.rl.dqn.replay_buffer import ReplayBuffer


"""
YETI

- time env
- checkpoint save & load

additions:

- soft update
- dueling
- double
- prio?2
"""


class DQN(object):

    @store_args
    def __init__(self, env, network_type='mlp', max_timesteps=5e7, batch_size=32, buffer_size=1e6, eta=1e-3, gamma=0.99,
                 final_eps=0.05, anneal_eps_until=1e6, training_start=1e5, target_update_freq=1e4, render_training=False,
                 optimizer=tf.train.AdamOptimizer, debug=False, use_tensorboard=True, tb_dir='/tmp/tf/', rolling_n=50,
                 reward_clipping=False, clean_tb=False):

        self.logger = logging.getLogger(__name__)

        # whether to use tensorboard or not
        self._tb = use_tensorboard

        # env specific parameter
        obs_shape = env.observation_space.shape
        num_actions = env.action_space.n

        # tf scopes
        self.Q_SCOPE = 'q_network'
        self.TARGET_SCOPE = 'target_network'

        # init q network, target network, replay buffer
        self.q_net, self.q_in = build_network(obs_shape, num_actions, network_type=network_type, name=self.Q_SCOPE)
        self.target_net, self.target_in = build_network(obs_shape, num_actions, network_type=network_type,
                                                        name=self.TARGET_SCOPE)
        self.replay_buffer = ReplayBuffer(buffer_size)

        # save variables from q and target network for
        self.q_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Q_SCOPE)
        self.target_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.TARGET_SCOPE)

        # define additional variables used in loss function
        self._L_r = tf.placeholder(tf.float32, (None,), name='loss_rewards')
        self._L_a = tf.placeholder(tf.int32, (None,), name='loss_actions')
        self._L_d_inv = tf.placeholder(tf.float32, (None,), name='loss_inverted_dones')

        # epsilon schedule
        self.eps = LinearSchedule(max_t=self.anneal_eps_until, final=final_eps)

        # init optimizer with loss to minimize
        self.opt = self.optimizer(learning_rate=self.eta)
        self.train_op = self.opt.minimize(self._loss(), var_list=self.q_net_vars)

        # global tf.Session and Graph init
        self.sess = tf.Session()

        # launch debug session if requested
        if self.debug:
            self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "localhost:6064")

        # collect summaries
        if self._tb:
            # more placeholders for summarised variables; along with summaries
            self.eps_ph = tf.placeholder(tf.float32, (), name='epsilon')
            self.rew_ph = tf.placeholder(tf.float32, (), name='rolling-reward')

            scalar_summary('epsilon', self.eps_ph)
            scalar_summary('reward', self.rew_ph)

            # display q_values while training
            for a_i in range(num_actions):
                scalar_summary('QTa_{}'.format(a_i + 1), tf.reduce_mean(self.target_net[:, a_i]), scope='Q-Values')
                scalar_summary('Qa_{}'.format(a_i+1), tf.reduce_mean(self.q_net[:, a_i]), scope='Q-Values')

            # plot network weights
            with tf.variable_scope('weights'):
                for qv in self.q_net_vars: tf.summary.histogram('{}'.format(qv.name), qv)
                for tv in self.target_net_vars: tf.summary.histogram('{}'.format(tv.name), tv)

            # gradient histograms
            grads = self.opt.compute_gradients(self._loss(tb=False))
            with tf.variable_scope('gradients'):
                for g in grads: tf.summary.histogram('{}-grad'.format(g[1].name), g[0])

            self.merge_op = tf.summary.merge_all()

            # clean previous runs or add new one
            if not self.clean_tb:
                rename_latest_run(self.tb_dir)
            else:
                clean_dir(self.tb_dir)

            self.writer = tf.summary.FileWriter('{}/run-latest'.format(self.tb_dir), graph=tf.get_default_graph())

        # init variables etc.
        self._init_tf()

        # synchronize networks before learning
        self._update_target()

    def _init_tf(self):
        """ Initializes tensorflow stuff """

        self.sess.run(tf.global_variables_initializer())

    def _update_target(self):
        """ Synchronises weights from q network to target network """

        self.logger.debug('Updating target network')

        ops = []
        for q_var, t_var in zip(self.q_net_vars, self.target_net_vars):
           ops.append(t_var.assign(q_var))

        self.sess.run(ops)

    def _loss(self, tb=True):
        """ Defines loss as layed out in the original Nature paper """

        with tf.variable_scope('loss'):
            # calculate target
            y = self._L_r + (self.gamma * self._L_d_inv * tf.reduce_max(self.target_net, axis=1))

            # select q value of taken action
            qj = tf.gather(self.q_net, self._L_a, axis=1)

            # apply huber loss TODO try tf.losses.huber_loss as comparison
            l = huber_loss(y - qj)

        if tb:
            scalar_summary('target', tf.reduce_mean(y))
            scalar_summary('huber-loss', tf.reduce_mean(l))
            tf.summary.histogram('selected_Q', qj)

        return l

    def learn(self):
        """ Learns Q function for a given amount of timesteps """

        # reset env, store first observation
        obs = self.env.reset()

        # save all episode rewards
        episode_reward_series = [[0.0]]
        episode_rewards = []

        self.logger.info('Starting Exploration')
        for t in range(int(self.max_timesteps)):

            # decide on action either by policy or chose a random one TODO decrease eps after training start?
            epsilon = self.eps.value(t)
            _rand = np.random.choice([True, False], p=[epsilon, 1-epsilon])
            if _rand:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.sess.run(self.q_net, {self.q_in: [obs]}), axis=1)
                assert len(action) == 1, 'only one action can be taken!'
                action = action[0]

            # take action in environment
            new_obs, reward, done, _ = self.env.step(action)

            # clip reward
            if self.reward_clipping:
                reward = 1 if reward > 0 else -1 if reward < 0 else 0

            # append current rewards to episode reward series
            episode_reward_series[-1].append(reward)

            # store new transition
            self.replay_buffer.add(obs, action, reward, new_obs, float(done))

            # t + 1 -> t
            obs = new_obs

            if self.render_training:
                self.env.render()

            if t == self.training_start:
                self.logger.info('Training starts now! (t = {})'.format(t))

            # final calculations and env reset
            if done:
                # calculate total reward
                episode_rewards.append(np.sum(episode_reward_series[-1]))
                episode_reward_series.append([0.0])

                obs = self.env.reset()

            if t >= self.training_start:

                # calculate rolling reward
                rr = np.mean(episode_rewards[-self.rolling_n:]) if len(episode_rewards) > 0 else 0.0

                # small episode results table
                if done:
                    result_table = [
                        ['reward', episode_rewards[-1]],
                        ['epsilon', epsilon]
                    ]
                    print('    EPISODE {}\n{}'.format(len(episode_rewards), tabulate(result_table)))

                o, a, r, no, d = self.replay_buffer.sample(self.batch_size)

                # invert dones => 0 for done will zero out gamma * Q(o_t+1, a')
                d_inv = np.array([1. if x == 0. else 0. for x in d])

                # collect summaries if needed or just train
                if self._tb:
                    summary, _ = self.sess.run([self.merge_op, self.train_op],
                                               feed_dict={self.target_in: no, self.q_in: o, self._L_r: r, self._L_a: a,
                                                          self._L_d_inv: d_inv, self.eps_ph: epsilon,
                                                          self.rew_ph: rr})
                    self.writer.add_summary(summary, t)
                else:
                    self.sess.run(self.train_op, feed_dict={self.target_in: no, self.q_in: o, self._L_r: r,
                                                            self._L_a: a, self._L_d_inv: d_inv})

                # sync target network every C steps
                if (t - self.training_start) % self.target_update_freq == 0:
                    self._update_target()

        # total reward of last (possibly interrupted) episode
        episode_rewards.append(np.sum(episode_reward_series[-1]))

    def run(self):
        """ Runs policy on given environment """
        pass


if __name__ == '__main__':
    import gym

    env = gym.make('CartPole-v0')
    agent = DQN(env, buffer_size=5e3, training_start=5e3, target_update_freq=1e2, anneal_eps_until=1e4,
                eta=15e-3, clean_tb=True)
    agent.learn()
