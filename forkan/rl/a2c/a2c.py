import numpy as np
import tensorflow as tf

from forkan.rl import BaseAgent
from forkan.common.policies import build_policy
from forkan.common.utils import discount_with_dones
from forkan.common.tf_utils import scalar_summary, entropy_from_logits

from tabulate import tabulate

"""
TODO's

- noise on logits?
"""


class A2C(BaseAgent):

    def __init__(self,
                 env,
                 alg_name='a2c',
                 name='default',
                 policy_type='mini-mlp',
                 total_timesteps=5e7,
                 lr=1e-3,
                 entropy_coef=0.01,
                 gamma=0.99,
                 tmax=5,
                 v_loss_coef=0.5,
                 max_grad_norm=None,
                 reward_clipping=None,
                 solved_callback=None,
                 print_freq=None,
                 render_training=False,
                 **kwargs,
                 ):
        """
        Implementation of the Deep Q Learning (DQN) algorithm formulated by Mnih et. al.
        Contains some well known improvements over the vanilla DQN.

        Parameters
        ----------
        env: gym.Environment
            (gym) Environment the agent shall learn from and act on

        alg_name: str
            name of algorithm, e.g. a2c, dqn, trpo etc

        name: str
            descriptive name of this DQN configuration, e.g. 'atari-breakout'

        policy_type: str
            which policy from common.policies is loaded

        total_timesteps: int or float
            number of training timesteps

        lr: float
            learning rate

        entropy_coef: float
            entropy coefficient in policy loss

        gamma: float
            discount factor gamma for bellman target

        tmax: int
            number of steps the agent acts on environment per timestep

        v_loss_coef: float
            value function loss coefficient in final loss

        max_grad_norm: int
            if not None, gradients are clipped by this value by norm

        reward_clipping: float
            rewards will be clipped to this value if not None

        solved_callback: function
           function which gets as an input the episode rewards as an array and must return a bool.
           if returned True, the training is considered as done and therefore prematurely interrupted.

        print_freq: int
            prints status every x episodes to stdout

        render_training: bool
            whether to render the environment while training

        """

        self.policy_type = policy_type

        # hyperparameter
        self.total_timesteps = total_timesteps
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.gamma = gamma

        self.tmax = tmax
        self.v_loss_coef = v_loss_coef
        self.max_grad_norm = max_grad_norm
        self.reward_clipping = reward_clipping

        self.solved_callback = solved_callback
        self.print_freq = print_freq
        self.render_training = render_training

        super().__init__(env, alg_name, name, **kwargs)

        # number of environments
        self.num_envs = env.num_envs
        self.batch_size = self.num_envs * self.tmax

        # env specific parameter
        self.obs_shape = env.observation_space.shape
        self.num_actions = env.action_space.n

        # scope name for policy tensors
        self.policy_scope = 'policy'

        # create value net, action net and policy output (and get input placeholder)
        self.obs_ph, self.logits, self.values, self.action = build_policy(self.obs_shape, self.num_actions, scope=self.policy_scope,
                                                             policy_type=self.policy_type, reuse=False)

        # store list of policy network variables
        def _get_trainable_variables(scope):
            with tf.variable_scope(scope):
                return tf.trainable_variables()

        self.policy_net_vars = _get_trainable_variables(scope=self.policy_scope)

        # setup placeholders for later
        self.adv_ph = tf.placeholder(tf.float32, (None, ), name='advantage-values')
        self.actions_ph = tf.placeholder(tf.int32, (None, ), name='actions')
        self.dis_rew_ph = tf.placeholder(tf.float32, (None, ), name='discounted-reward')

        # construct policy loss
        self.oh_actions = tf.one_hot(self.actions_ph, self.num_actions)
        self.neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.oh_actions, logits=self.logits)
        self.pi_loss = tf.reduce_mean(self.neglogp * self.adv_ph)

        self.pi_entropy = tf.reduce_mean(entropy_from_logits(self.logits))
        self.policy_loss = self.pi_loss + self.entropy_coef * self.pi_entropy

        # construct value loss
        self.value_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.expand_dims(self.dis_rew_ph, -1),
                                                                      predictions=self.values))
        # final loss
        self.loss = self.policy_loss + self.v_loss_coef * self.value_loss

        # create optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        # specify loss function, only include Q network variables for gradient computation
        self.gradients = self.opt.compute_gradients(self.loss, var_list=self.policy_net_vars)

        # clip gradients by norm
        if self.max_grad_norm is not None:
            for idx, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)

        # create training op
        self.train_op = self.opt.apply_gradients(self.gradients)

        # takes care of tensorboard, debug and checkpoint init
        self._finalize_init()

    def _setup_tensorboard(self):
        """
        Adds all variables that might help debugging to Tensorboard.
        At the end, the FileWriter is constructed pointing to the specified directory.

        """

        self.logger.info('Saving Tensorboard summaries to {}'.format(self.tensorboard_dir))

        self.ret_ph = tf.placeholder(tf.float32, (), name='mean-return')

        scalar_summary('mean-return', self.ret_ph)

        with tf.variable_scope('loss'):
            scalar_summary('loss', self.loss)
            scalar_summary('value-loss', self.value_loss)
            scalar_summary('policy-loss', self.policy_loss)
            scalar_summary('policy-entropy', self.pi_entropy)

        with tf.variable_scope('value'):
            scalar_summary('value_target', tf.reduce_mean(self.dis_rew_ph))
            scalar_summary('value', tf.reduce_mean(self.values))

        # plot network weights
        with tf.variable_scope('weights'):
            for pv in self.policy_net_vars: tf.summary.histogram('{}'.format(pv.name), pv)

        # gradient histograms
        with tf.variable_scope('gradients'):
            for g in self.gradients: tf.summary.histogram('{}-grad'.format(g[1].name), g[0])

    def _gradient_fd(self, dis_rew, obses, logis, actions, values, mean_ret=None):
        """ Takes multistep-batch and returns feed dict for gradient computation. """

        # calculate advantage values
        advs = (dis_rew - np.squeeze(values))

        # we assign values to defined tensors here so we
        # avoid one additional forward pass
        _fd = {
            self.adv_ph: advs,
            self.actions_ph: actions,
            self.obs_ph: obses,
            self.dis_rew_ph: dis_rew,
            self.logits: logis,
            self.values: values,
        }

        if self.use_tensorboard:
            _fd[self.ret_ph] = mean_ret

        return _fd

    def value(self, obs):
        """ Returns batched values of obs. """
        return self.sess.run([self.values], feed_dict={
                    self.obs_ph: obs
                })[0]


    def learn(self):
        """ Learns Q function for a given amount of timesteps. """

        # reset env, store first observation
        obs_t = self.env.reset()
        d_t = [False] * self.num_envs

        # real timesteps
        T = 0

        # some statistics
        cur_eps_rets = np.zeros((self.num_envs,), np.float32)
        past_returns = []
        mean_ret = 0.0

        self.logger.info('Starting training!')

        while T < self.total_timesteps:


            # We initialize the lists that will contain the mb of experiences
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logi = [], [], [], [], [], []

            # perform multisteps
            for n in range(self.tmax):

                # calculate pi & state value
                logi, val, acs = self.sess.run([self.logits, self.values, self.action], feed_dict={
                    self.obs_ph: obs_t
                })

                # observe o+1, r+1
                obs_tp1, r_t, d_tp1, _ = self.env.step(acs)

                # Append the experiences
                mb_obs.append(np.copy(obs_t))
                mb_actions.append(acs)
                mb_logi.append(logi)
                mb_values.append(val)
                mb_dones.append(d_t)
                mb_rewards.append(r_t)

                # TODO WHY
                for n, done in enumerate(d_t):
                    if done:
                        obs_t[n] = obs_t[n] * 0

                # t <-- t + 1
                T += 1
                obs_t = obs_tp1
                d_t = d_tp1

                # we check for each env whether it finised, otherwise store rewards
                # resets are not needed, they happen in the threads after episode termination
                for n, (re, do) in enumerate(zip(r_t, d_t)):

                    # accumulate reward for this step
                    cur_eps_rets[n] += re

                    if do:
                        # store final reward in list
                        past_returns.append(cur_eps_rets[n])

                        # zero reward
                        cur_eps_rets[n] *= 0

                        # calculate mean returns
                        if len(past_returns) > 20:
                            mean_ret = np.mean(past_returns[-20:])

                        if self.print_freq is not None and len(past_returns) % self.print_freq == 0:
                            result_table = [
                                ['T', T],
                                ['episode', len(past_returns)//self.num_envs],
                                ['mean return', mean_ret],
                            ]

                            print('\n{}'.format(tabulate(result_table)))

            # TODO why additional done append?
            mb_dones.append(d_t)

            # Batch of steps to batch of rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape((self.batch_size, ) + self.obs_shape)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
            mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
            mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
            mb_logi = np.asarray(mb_logi, dtype=np.float32).swapaxes(1, 0)
            mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
            mb_masks = mb_dones[:, :-1]
            mb_dones = mb_dones[:, 1:]

            if self.gamma > 0.0:
                # calculate pi & state value
                _, last_values = self.sess.run([self.logits, self.values], feed_dict={
                    self.obs_ph: obs_t
                })
                for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                    rewards = rewards.tolist()
                    dones = dones.tolist()
                    if dones[-1] == 0:
                        rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                    else:
                        rewards = discount_with_dones(rewards, dones, self.gamma)

                    mb_rewards[n] = rewards

            mb_actions = mb_actions.reshape((self.batch_size, ))
            mb_logi = mb_logi.reshape((self.batch_size, self.num_actions))

            mb_rewards = mb_rewards.flatten()
            mb_values = np.expand_dims(mb_values.flatten(), -1)
            mb_masks = mb_masks.flatten()

            # build feed dict for gradient and/or training operation
            g_feed = self._gradient_fd(dis_rew=mb_rewards,
                                       obses=mb_obs,
                                       logis=mb_logi,
                                       actions=mb_actions,
                                       values=mb_values,
                                       mean_ret=mean_ret)

            # run training step using already computed data
            if self.use_tensorboard:
                _, sum = self.sess.run([self.train_op, self.merge_op], feed_dict=g_feed)
                self.writer.add_summary(sum, T)
            else:
                self.sess.run(self.train_op, feed_dict=g_feed)

        # finalize training, e.g. set flags, write done-file
        self._finalize_training()

    def run(self, render=True):
        """ Runs policy on given environment """

        pass
