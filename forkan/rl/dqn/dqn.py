import os
import logging
import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug
from tabulate import tabulate

from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from forkan import weights_path
from forkan.common.utils import rename_latest_run, clean_dir, create_dir
from forkan.common.tf_utils import scalar_summary
from forkan.rl.dqn.networks import build_network


class DQN(object):

    def __init__(self,
                 env,
                 name='default',
                 network_type='mini-mlp',
                 total_timesteps=5e7,
                 batch_size=32,
                 lr=1e-3,
                 gamma=0.99,
                 buffer_size=1e6,
                 final_eps=0.05,
                 exploration_fraction=0.1,
                 training_start=1e5,
                 target_update_freq=1e4,
                 optimizer=tf.train.AdamOptimizer,
                 gradient_clipping=None,
                 reward_clipping=False,
                 tau=1.,
                 double_q=False,
                 dueling=False,
                 prioritized_replay=False,
                 prioritized_replay_alpha=0.5,
                 prioritized_replay_beta_init=0.4,
                 prioritized_replay_beta_fraction=1.0,
                 prioritized_replay_eps=1e-6,
                 rolling_reward_mean=20,
                 solved_callback=None,
                 render_training=False,
                 debug=False,
                 use_tensorboard=True,
                 tensorboard_dir='/tmp/tensorboard/dqn/',
                 tensorboard_suffix=None,
                 clean_tensorboard_runs=False,
                 use_checkpoints=True,
                 clean_previous_weights=False,
                 ):
        """
        Implementation of the Deep Q Learning (DQN) algorithm formulated by Mnih et. al.
        Contains some well known improvements over the vanilla DQN.

        Parameters
        ----------
        env: gym.Environment
            (gym) Environment the agent shall learn from and act on

        name: str
            descriptive name of this DQN configuration, e.g. 'atari-breakout'

        network_type: str
            which network is from 'networks.py'

        total_timesteps: int or float
            number of training timesteps

        batch_size: int
            size of minibatch per backprop

        lr: float
            learning rate

        gamma: float
            discount factor gamma for bellman target

        buffer_size: int or float
            maximum number of in replay buffer

        final_eps: float
            value to which epsilon is annealed

        exploration_fraction: float
            fraction of traing timesteps over which epsilon is annealed

        training_start: int
            timestep at which training of the q network begins

        target_update_freq: int
            frequency of target network updates (in timesteps)

        optimizer: tf.Optimizer
            optimizer class which shall be used such as Adam or RMSprop

        gradient_clipping: int
            if not None, gradients are clipped by this value by norm

        reward_clipping: float
            rewards will be clipped to this value if not None

        tau: float
            interpolation constant for soft update. 1.0 corresponds to
            a full synchronisation of networks weights, as in the original DQN paper

        double_q: bool
            enables Double Q Learning for DQN

        dueling: bool
            splits network architecture into advantage and value streams. V(s, a) gets
            more frequent updates, should stabalize learning

        prioritized_replay: True
            use (proportional) prioritized replay

        prioritized_replay_alpha: float
            alpha for weighting priorization

        prioritized_replay_beta_init: float
            initial value of beta for prioritized replay buffer

        prioritized_replay_beta_fraction: float
            fraction of total timesteps to anneal beta to 1.0

        prioritized_replay_eps: float
            epsilon to add to the TD errors when updating priorities.

        rolling_reward_mean: int
            window of which the rolling mean in the statistics is computed

        solved_callback: function
            function which gets as an input the episode rewards as an array and must return a bool.
            if returned True, the training is considered as done and therefore prematurely interrupted.

        render_training: bool
            whether to render the environment while training

        debug: bool
            if true, a TensorBoard debugger session is started

        use_tensorboard: bool
            toggles TensorBoard support. If enabled, variable summaries are created and
            written to disk in real time while training.

        tensorboard_dir: str
            Parent directory to save the TensorBoard files to. Within this directory, a new folder is
            created for every training run of the policy. Folders are named as 'run-X' or 'run-latest',
            where X stands for the runs ID.

        tensorboard_suffix: str
            Addition to the foldername for individual runs, could, for example, contain information about
            hyperparamets used. Foldernames will be of the form 'run-SUFFIX-ID'.

        clean_tensorboard_runs: bool
            If true, data of other runs is wiped before execution. This exists mainly to avoid
            disk bloating when testing a lot.

        use_checkpoints: bool
            Saves the model after each episode and upon every policy improvement. A csv-file
            is also written to disk alongside the weights containing information about the run.

        clean_previous_weights: bool
            If true, weights of other runs is wiped before execution. This exists mainly to avoid
            disk bloating when testing a lot.

        """

        # instance name
        self.name = name

        # environment to act on / learn from
        self.env = env

        # basic DQN parameters
        self.total_timesteps = float(total_timesteps)
        self.buffer_size = int(float(buffer_size))
        self.batch_size = batch_size
        self.final_eps = final_eps
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.exploration_fraction = float(exploration_fraction)
        self.training_start = int(float(training_start))
        self.target_update_freq = int(float(target_update_freq))

        # tf.Optimizer
        self.optimizer = optimizer

        # minor changes as suggested in some papers
        self.gradient_clipping = int(gradient_clipping) if gradient_clipping is not None else None
        self.reward_clipping = int(reward_clipping) if reward_clipping is not None else None

        # enhancements to DQN published in papers
        self.tau = float(tau)
        self.double_q = double_q
        self.dueling = dueling
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_alpha = float(prioritized_replay_alpha)
        self.prioritized_replay_beta_init = float(prioritized_replay_beta_init)
        self.prioritized_replay_beta_fraction = float(prioritized_replay_beta_fraction)
        self.prioritized_replay_eps = float(prioritized_replay_eps)

        # function to determine whether agent is able to act well enough
        self.solved_callback = solved_callback

        # call env.render() each training step
        self.render_training = render_training

        # rolling mean reward window
        self.rolling_reward_mean = rolling_reward_mean

        # tensorboard and debug related variables
        self.debug = debug
        self.use_tensorboard = use_tensorboard
        self.tensorboard_dir = tensorboard_dir
        self.clean_tensorboard_runs = clean_tensorboard_runs
        self.tensorboard_suffix = tensorboard_suffix

        # checkpoint
        self.use_checkpoints = use_checkpoints
        self.clean_previous_weights = clean_previous_weights

        # calculate timestep where epsilon reaches its final value
        self.schedule_timesteps = int(self.total_timesteps * self.exploration_fraction)

        # concat name of instance to path -> distinction between saved instances
        self.checkpoint_dir = '{}/dqn/{}/'.format(weights_path, name)

        # sanity checks
        assert 0.0 < self.tau <= 1.0

        # logger for different levels
        self.logger = logging.getLogger(__name__)

        # env specific parameter
        self.obs_shape = env.observation_space.shape
        self.num_actions = env.action_space.n

        # tf scopes
        self.Q_SCOPE = 'q_network'
        self.TARGET_SCOPE = 'target_network'

        # build Q and target network; using different scopes to distinguish variables for gradient computation
        self.q_t, self.q_t_in = build_network(self.obs_shape, self.num_actions, network_type=network_type,
                                              dueling=self.dueling, scope=self.Q_SCOPE, summaries=True)
        self.target_tp1, self.target_tp1_in = build_network(self.obs_shape, self.num_actions, dueling=self.dueling,
                                                            network_type=network_type, scope=self.TARGET_SCOPE)

        # double Q learning needs to pass observations t+1 to the q networks for action selection
        # so we reuse already created q network variables but with different input
        if self.double_q:
            self.q_tp1, self.q_tp1_in = build_network(self.obs_shape, self.num_actions, dueling=self.dueling,
                                                      network_type=network_type, scope=self.Q_SCOPE, reuse=True)

        # create replay buffer
        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, self.prioritized_replay_alpha)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)

        # list of variables of the different networks. required for copying
        # Q to target network and excluding target network variables from backprop
        self.q_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Q_SCOPE)
        self.target_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.TARGET_SCOPE)

        # placeholders used in loss function
        self._L_r = tf.placeholder(tf.float32, (None,), name='loss_rewards')
        self._L_a = tf.placeholder(tf.int32, (None,), name='loss_actions')
        self._L_d = tf.placeholder(tf.float32, (None,), name='loss_dones')

        # pointer to td error vector
        self._td_errors = tf.placeholder(tf.float32, (None,), name='td_errors')

        # configure prioritized replay
        if self.prioritized_replay:
            self._is_weights = tf.placeholder(tf.float32, (None,), name='importance_sampling_weights')

            # schedule for PR beta
            beta_steps = int(self.total_timesteps * self.prioritized_replay_beta_fraction)
            self.pr_beta = LinearSchedule(beta_steps, initial_p=prioritized_replay_beta_init, final_p=1.0)

        # epsilon schedule
        self.eps = LinearSchedule(self.schedule_timesteps, final_p=final_eps)

        # init optimizer
        self.opt = self.optimizer(self.lr)

        # specify loss function, only include Q network variables for gradient computation
        self.gradients = self.opt.compute_gradients(self._loss(), var_list=self.q_net_vars)

        # clip gradients by norm
        if self.gradient_clipping is not None:
            for idx, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[idx] = (tf.clip_by_norm(grad, self.gradient_clipping), var)

        # create training op
        self.train_op = self.opt.apply_gradients(self.gradients)

        # update_target_fn will be called periodically to copy Q network to target Q network
        # variable lists are sorted by name to ensure that correct values are copied
        self.update_target_ops = []
        for var_q, var_target in zip(sorted(self.q_net_vars, key=lambda v: v.name),
                                   sorted(self.target_net_vars, key=lambda v: v.name)):
            v_update = var_target.assign(self.tau * var_q + (1 - self.tau) * var_target)
            self.update_target_ops.append(v_update)
        self.update_target_ops = tf.group(*self.update_target_ops)

        # global tf.Session and Graph init
        self.sess = tf.Session()

        # launch debug session
        if self.debug:
            self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "localhost:6064")

        # create tensorboard summaries
        if self.use_tensorboard:
            self._setup_tensorboard()

        # init variables
        self.sess.run(tf.global_variables_initializer())

        # sync networks before training
        self.sess.run(self.update_target_ops)

        # flag indicating whether this instance is completely trained
        self.is_trained = False

        # if this instance is working with checkpoints, we'll check whether
        # one is already there. if so, we continue training from that checkpoint,
        # i.e. load the saved weights into target and online network.
        if self.use_checkpoints:

            # remove old weights if needed and not already trained until the end
            if self.clean_previous_weights and not os.path.isfile('{}/done'.format(self.checkpoint_dir)):
                clean_dir(self.checkpoint_dir)

            # be sure that the directory exits
            create_dir(self.checkpoint_dir)

            # Saver objects handles writing and reading protobuf weight files
            self.saver = tf.train.Saver(var_list=tf.all_variables())

            # file handle for writing episode summaries
            self.csvlog = open('{}/progress.csv'.format(self.checkpoint_dir), 'a')

            # write headline if file is not empty
            if not os.stat('{}/progress.csv'.format(self.checkpoint_dir)).st_size == 0:
                self.csvlog.write('episode, epsilon, reward\n')

            # load already saved weights
            self._load()

    def __del__(self):
        """ Cleanup after object finalization """

        # close tf.Session
        if hasattr(self, 'sess'):
           self.sess.close()

        # close filehandle
        if hasattr(self, 'csvlog'):
            self.csvlog.close()

    def _setup_tensorboard(self):
        """
        Adds all variables that might help debugging to Tensorboard.
        At the end, the FileWriter is constructed pointing to the specified directory.

        """

        # more placeholders for summarised variables; along with summaries
        self.eps_ph = tf.placeholder(tf.float32, (), name='epsilon')
        self.rew_ph = tf.placeholder(tf.float32, (), name='rolling-reward')

        scalar_summary('epsilon', self.eps_ph)
        scalar_summary('reward', self.rew_ph)

        # display q_values while training
        for a_i in range(self.num_actions):
            scalar_summary('QTa_{}'.format(a_i + 1), tf.reduce_mean(self.target_tp1[:, a_i]), scope='Q-Values')
            scalar_summary('Qa_{}'.format(a_i + 1), tf.reduce_mean(self.q_t[:, a_i]), scope='Q-Values')

        # plot network weights
        with tf.variable_scope('weights'):
            for qv in self.q_net_vars: tf.summary.histogram('{}'.format(qv.name), qv)
            for tv in self.target_net_vars: tf.summary.histogram('{}'.format(tv.name), tv)

        # gradient histograms
        with tf.variable_scope('gradients'):
            for g in self.gradients: tf.summary.histogram('{}-grad'.format(g[1].name), g[0])

        # this operation can be run in a tensorflow session and will return all summaries
        # created above.
        self.merge_op = tf.summary.merge_all()

        # clean previous runs or add new one
        if not self.clean_tensorboard_runs:
            rename_latest_run(self.tensorboard_dir)
        else:
            clean_dir(self.tensorboard_dir)

        # if there is a directory suffix given, it will be included before the run number in the filename
        tb_dir_suffix = '' if self.tensorboard_suffix is None else '-{}'.format(self.tensorboard_suffix)
        self.writer = tf.summary.FileWriter('{}/run{}-latest'.format(self.tensorboard_dir, tb_dir_suffix),
                                            graph=tf.get_default_graph())

    def _loss(self):
        """ Defines loss as layed out in the original Nature paper """

        with tf.variable_scope('loss'):

            # either use maximum target q or use value from target network while the action is chosen by the q net
            if self.double_q:
                act_tp1_idxs = tf.stop_gradient(tf.argmax(self.q_tp1, axis=1))
                q_tp1 = tf.reduce_sum(self.target_tp1 * tf.one_hot(act_tp1_idxs, self.num_actions), axis=1)
            else:

                q_tp1 = tf.reduce_max(self.target_tp1, axis=1)

            # bellman target
            y = self._L_r + (self.gamma * (1.0 - self._L_d) * q_tp1)

            # select q value of taken action
            qj = tf.reduce_sum(self.q_t * tf.one_hot(self._L_a, self.num_actions), axis=1)

            # TD errors
            self._td_errors = qj - y

            # apply huber loss
            loss = tf.losses.huber_loss(y, qj)

        if self.use_tensorboard:
            scalar_summary('target', tf.reduce_mean(y))
            scalar_summary('huber-loss', tf.reduce_mean(loss))
            tf.summary.histogram('selected_Q', qj)

        #  importance sampling weights
        if self.prioritized_replay:
            updates = tf.reduce_mean(self._is_weights * loss)
        else:
            updates = tf.reduce_mean(loss)

        return updates

    def _build_feed_dict(self, obs_t, ac_t, rew_t, obs_tp1, dones, eps, rolling_rew, weights=None):
        """ Takes minibatch and returns feed dict for a tf.Session based on the algorithms configuration. """

        # first, add data required in all DQN configs
        feed_d = {
            self.q_t_in: obs_t,
            self.target_tp1_in: obs_tp1,
            self._L_r: rew_t,
            self._L_a: ac_t,
            self._L_d: dones
        }

        # pass obs t+1 to q network
        if self.double_q:
            feed_d[self.q_tp1_in] = obs_tp1

        # importance sampling weights
        if self.prioritized_replay:
            feed_d[self._is_weights] = weights

        # variables only necessary for TensorBoard visualisation
        if self.use_tensorboard:
            feed_d[self.eps_ph] = eps
            feed_d[self.rew_ph] = rolling_rew

        return feed_d

    def _save(self, weight_dir='latest'):
        """ Saves current weights under CHECKPOINT_DIR/weight_dir/ """

        self.saver.save(self.sess, '{}/{}/'.format(self.checkpoint_dir, weight_dir))

    def _load(self):
        """
        Loads model weights. If the done-file exists, we know that
        training finished for this set of weights, so we

        """
        # check whether the model being loaded was fully trained
        if os.path.isfile('{}/done'.format(self.checkpoint_dir)):
            self.logger.debug('Loading finished weights!')
            self.saver.restore(self.sess, '{}/best/'.format(self.checkpoint_dir))

            # set model as trained
            self.is_trained = True
        elif os.path.isdir('{}/latest/'.format(self.checkpoint_dir)):
            self.logger.warning('Loading pre-trained weights. As this model is not marked as \'done\', \n' +
                                'training will start from t=0 using these weights (this includes filling \n' +
                                'the replay buffer). Make sure to have a solved_callback specified to \n' +
                                'avoid training a good policy for too long.')
            self.saver.restore(self.sess, '{}/latest/'.format(self.checkpoint_dir))
        else:
            self.logger.debug('No weights to load found!')

    def _finalize_training(self):
        """ Takes care of things once training finished """

        # set model as trained
        self.is_trained = True

        # create done-file
        with open('{}/done'.format(self.checkpoint_dir), 'w'): pass

        # close file handle to csv log file
        self.csvlog.close()

    def learn(self):
        """ Learns Q function for a given amount of timesteps """

        # reset env, store first observation
        obs_t = self.env.reset()

        # save all episode rewards
        episode_reward_series = [[0.0]]
        episode_rewards = []

        self.logger.info('Starting Exploration')

        for t in range(int(self.total_timesteps)):

            # decide on action either by policy or chose a random one
            epsilon = self.eps.value(t)
            _rand = np.random.choice([True, False], p=[epsilon, 1-epsilon])
            if _rand:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.sess.run(self.q_t, {self.q_t_in: [obs_t]}), axis=1)
                assert len(action) == 1, 'only one action can be taken!'
                action = action[0]

            # act on environment with chosen action
            obs_tp1, reward, done, _ = self.env.step(action)

            # clip reward
            if self.reward_clipping:
                reward = 1 if reward > 0 else -1 if reward < 0 else 0

            # store new transition
            self.replay_buffer.add(obs_t, action, reward, obs_tp1, float(done))

            # new observation will be current one in next iteration
            obs_t = obs_tp1

            # append current rewards to episode reward series
            episode_reward_series[-1].append(reward)

            if self.render_training:
                self.env.render()

            if t == self.training_start:
                self.logger.info('Training starts now! (t = {})'.format(t))

            # final calculations and env reset
            if done:
                # calculate total reward
                episode_rewards.append(np.sum(episode_reward_series[-1]))
                episode_reward_series.append([0.0])

                # reset env to initial state
                obs_t = self.env.reset()

            # start training after warmup period
            if t >= self.training_start:

                # calculate rolling reward
                rolling_r = np.mean(episode_rewards[-self.rolling_reward_mean:]) if len(episode_rewards) > 0 else 0.0

                # post episode stuff: printing and saving
                if done:
                    result_table = [
                        ['t', t],
                        ['episode', len(episode_rewards)],
                        ['mean_reward [20]', rolling_r],
                        ['epsilon', epsilon]
                    ]
                    print('\n{}'.format(tabulate(result_table)))

                    # if the policy improved, save as new best
                    if len(episode_rewards) >= 2:
                        if episode_rewards[-1] > episode_rewards[-2]:
                            self._save('best')

                    # save latest policy
                    self._save()

                    # write current values to csv log
                    self.csvlog.write('{}, {}, {}\n'.format(len(episode_rewards), epsilon, episode_rewards[-1]))

                # sample batch of transitions randomly for training and build feed dictionary
                # prioritized replay needs a beta and returns weights.
                if self.prioritized_replay:
                    o_t, a_t, r_t, o_tp1, do, is_ws, batch_idxs = self.replay_buffer.sample(self.batch_size,
                                                                                            self.pr_beta.value(t))
                    feed = self._build_feed_dict(o_t, a_t, r_t, o_tp1, do, epsilon, rolling_r, weights=is_ws)
                else:
                    o_t, a_t, r_t, o_tp1, do = self.replay_buffer.sample(self.batch_size)
                    feed = self._build_feed_dict(o_t, a_t, r_t, o_tp1, do, epsilon, rolling_r)

                # run training (and summary) operations
                if self.use_tensorboard:
                    summary, _, td_errors = self.sess.run([self.merge_op, self.train_op, self._td_errors],
                                                          feed_dict=feed)
                    self.writer.add_summary(summary, t)
                else:
                    self.sess.run(self.train_op, feed_dict=feed)

                # new td errors needed to update buffer weights
                if self.prioritized_replay:
                    new_prios = np.abs(td_errors) + self.prioritized_replay_eps
                    self.replay_buffer.update_priorities(batch_idxs, new_prios)

                # sync target network every C steps
                if (t - self.training_start) % self.target_update_freq == 0:
                    self.sess.run(self.update_target_ops)

            if self.solved_callback is not None:
                if self.solved_callback(episode_rewards):
                    self.logger.info('Solved!')
                    break

        # total reward of last episode
        episode_rewards.append(np.sum(episode_reward_series[-1]))

        # finalize training, e.g. set flags, write done-file
        self._finalize_training()

    def run(self, render=True):
        """ Runs policy on given environment """

        if not self.is_trained:
            self.logger.warning('Trying to run untrained model!')

        # set necessary parameters to their defaults
        epsilon = self.final_eps
        reward = 0.0
        obs = self.env.reset()

        while True:

            # decide on action either by policy or chose a random one
            _rand = np.random.choice([True, False], p=[epsilon, 1 - epsilon])
            if _rand:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.sess.run(self.q_t, {self.q_t_in: [obs]}), axis=1)
                assert len(action) == 1, 'only one action can be taken!'
                action = action[0]

            # act on environment with chosen action
            obs, rew, done, _ = self.env.step(action)
            reward += rew

            if render:
                self.env.render()

            if done:
                self.logger.info('Done! Reward {}'.format(reward))
                reward = 0.0
                obs = self.env.reset()


if __name__ == '__main__':
    from forkan import ConfigManager

    cm = ConfigManager(['cart-dqn'])
    cm.exec()

