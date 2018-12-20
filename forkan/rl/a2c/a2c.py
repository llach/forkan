import numpy as np
import tensorflow as tf

from forkan.rl import BaseAgent
from forkan.common.policies import build_policy
from forkan.common.tf_utils import scalar_summary, entropy_from_logits

from tabulate import tabulate

"""
TODO's

- value loss VERY high => print value loss in openai impl ( maybe R is calculated wrong?)
- clipping with .5 leads to nan => use tf.gradients instead compute gradients (whats the difference?
                                                                    maybe Optimizer calcs already drive gradients 
                                                                    close to zero. clipping by norm drives values
                                                                    further towards numerically unstable 
                                                                    territory.??)

"""


class A2C(BaseAgent):

    def __init__(self,
                 env,
                 alg_name='a2c',
                 name='default',
                 policy_type='mini-mlp',
                 total_timesteps=5e7,
                 lr=1e-3,
                 beta=0.01,
                 gamma=0.99,
                 tmax=5,
                 v_loss_coef=0.5,
                 gradient_clipping=None,
                 reward_clipping=None,
                 solved_callback=None,
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

        beta: float
            entropy coefficient in policy loss

        gamma: float
            discount factor gamma for bellman target

        tmax: int
            number of steps the agent acts on environment per timestep

        v_loss_coef: float
            value function loss coefficient in final loss

         gradient_clipping: int
                    if not None, gradients are clipped by this value by norm

        reward_clipping: float
            rewards will be clipped to this value if not None

        solved_callback: function
           function which gets as an input the episode rewards as an array and must return a bool.
           if returned True, the training is considered as done and therefore prematurely interrupted.

        render_training: bool
            whether to render the environment while training

        """

        self.policy_type = policy_type

        # hyperparameter
        self.total_timesteps = total_timesteps
        self.lr = lr
        self.beta = beta
        self.gamma = gamma

        self.tmax = tmax
        self.v_loss_coef = v_loss_coef
        self.gradient_clipping = gradient_clipping
        self.reward_clipping = reward_clipping

        self.solved_callback = solved_callback
        self.render_training = render_training

        super().__init__(env, alg_name, name, **kwargs)

        # env specific parameter
        self.obs_shape = env.observation_space.shape
        self.num_actions = env.action_space.n

        # create value net, action net and policy output (and get input placeholder)
        self.obs_ph, self.logits, self.value = build_policy(self.obs_shape, self.num_actions, scope='policy',
                                                            policy_type=self.policy_type, reuse=False)

        # store list of policy network variables
        self.policy_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')

        # setup placeholders for later
        self.adv_ph = tf.placeholder(tf.float32, (None, 1), name='advantage-values')
        self.actions_ph = tf.placeholder(tf.int32, (None, 1), name='actions')
        self.R_j_ph = tf.placeholder(tf.float32, (None, 1), name='discounted-reward')

        # construct policy loss
        self.oh_actions = tf.one_hot(self.actions_ph, self.num_actions)
        self.neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.oh_actions, logits=self.logits)
        self.pi_loss = tf.reduce_mean(self.neglogp * self.adv_ph)

        self.pi_entropy = tf.reduce_mean(entropy_from_logits(self.logits))
        self.policy_loss = self.pi_loss + self.beta * self.pi_entropy

        # construct value loss
        self.value_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.R_j_ph, predictions=self.value))

        # final loss
        self.loss = self.policy_loss + self.v_loss_coef * self.value_loss

        # create optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        # specify loss function, only include Q network variables for gradient computation
        self.gradients = self.opt.compute_gradients(self.loss, var_list=self.policy_net_vars)

        # clip gradients by norm
        if self.gradient_clipping is not None:
            for idx, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[idx] = (tf.clip_by_norm(grad, self.gradient_clipping), var)

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

        scalar_summary('loss', self.loss)
        scalar_summary('mean-return', self.ret_ph)
        scalar_summary('policy-entropy', self.pi_entropy)
        scalar_summary('policy-loss', self.policy_loss)
        scalar_summary('value-lass', self.value_loss)
        scalar_summary('R', tf.reduce_mean(self.R_j_ph))

        # plot network weights
        with tf.variable_scope('weights'):
            for pv in self.policy_net_vars: tf.summary.histogram('{}'.format(pv.name), pv)

        # gradient histograms
        with tf.variable_scope('gradients'):
            for g in self.gradients: tf.summary.histogram('{}-grad'.format(g[1].name), g[0])

    def _gradient_fd(self, R_js, obses, logis, actions, values, steps_t, mean_ret=None):
        """ Takes multistep-batch and returns feed dict for gradient computation. """

        # transform shapes of inputs to match batch shape convention
        logis = np.reshape(logis, (steps_t, self.num_actions))
        actions = np.expand_dims(actions, 1)
        values = np.expand_dims(values, 1)

        # calculate advantage values
        advs = (R_js - values)

        # we assign values to defined tensors here so we
        # avoid one additional forward pass
        _fd = {
            self.adv_ph: advs,
            self.actions_ph: actions,
            self.obs_ph: obses,
            self.R_j_ph: R_js,
            self.logits: logis,
            self.value: values,
        }

        if self.use_tensorboard:
            _fd[self.ret_ph] = mean_ret

        return _fd

    def learn(self):
        """ Learns Q function for a given amount of timesteps """

        # reset env, store first observation
        obs_t = self.env.reset()

        # real timesteps
        T = 0

        episode_returns = []
        epi_ret = 0.0
        mean_ret = 0.0

        cur_ent = 0.0

        self.logger.info('Starting training!')

        while T < self.total_timesteps:

            obses = []
            logis = []
            actions = np.array([])
            rewards = np.array([])
            dones = np.array([])
            values = np.array([])

            steps_t = 0

            # perform multisteps
            for n in range(self.tmax):

                # calculate pi & state value
                logi, val = self.sess.run([self.logits, self.value], feed_dict={
                    self.obs_ph: [obs_t]
                })

                # chose action based on highest action
                action = np.argmax(logi, axis=1)[0]

                # observe o+1, r+1
                obs_tp1, r_t, d_t, _ = self.env.step(action)

                # store data
                logis.append(logi)
                obses.append(obs_t)
                actions = np.append(actions, [action])
                rewards = np.append(rewards, r_t)
                dones = np.append(dones, d_t)
                values = np.append(values, [val])

                # t <-- t + 1
                T += 1
                steps_t += 1
                obs_t = obs_tp1

                epi_ret += r_t

                # we break if episode ended
                if d_t:
                    obs_t = self.env.reset()
                    episode_returns.append(epi_ret)
                    epi_ret = 0.0

                    if len(episode_returns) > 20:
                        mean_ret = np.mean(episode_returns[-20:])

                    result_table = [
                        ['T', T],
                        ['episode', len(episode_returns)],
                        ['mean return', mean_ret],
                        ['policy entropy', cur_ent],
                    ]
                    # print('\n{}'.format(tabulate(result_table)))

                    break

            # set  to either zero or the value of the last multistep state
            R = 0 if dones[-1] else values[-1]

            R_j = []
            for i in range(steps_t):
                if i == 0:
                    R_j.append([rewards[i] + self.gamma * R])
                else:
                    R_j.append([rewards[i] + self.gamma * R_j[-1][0]])

            # build feed dict for gradient and/or training operation
            g_feed = self._gradient_fd(R_js=R_j,
                                       obses=obses,
                                       logis=logis,
                                       actions=actions,
                                       values=values,
                                       steps_t=steps_t,
                                       mean_ret=mean_ret)

            # run training step using already computed data
            if self.use_tensorboard:
                _, cur_ent, sum = self.sess.run([self.train_op, self.pi_entropy, self.merge_op], feed_dict=g_feed)
                self.writer.add_summary(sum, T)
            else:
                _, cur_ent = self.sess.run([self.train_op, self.pi_entropy], feed_dict=g_feed)

        # finalize training, e.g. set flags, write done-file
        self._finalize_training()

    def run(self, render=True):
        """ Runs policy on given environment """

        pass
