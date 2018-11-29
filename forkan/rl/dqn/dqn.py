import os
import logging
import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug
from tabulate import tabulate

from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer

from forkan.common.utils import rename_latest_run, clean_dir, create_dir
from forkan.common.tf_utils import scalar_summary
from forkan.rl.dqn.networks import build_network


"""
YETI

enhacements:

- soft update
- dueling
- double
- prioritized replay

"""


class DQN(object):

    def __init__(self,
                 env,
                 name='default',
                 network_type='mlp',
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
                 checkpoint_dir='/tmp/tf-checkpoints/dqn/',
                 ):

        self.clean_previous_weights = clean_previous_weights
        self.solved_callback = solved_callback
        self.name = name
        self.use_checkpoints = use_checkpoints

        self.env = env
        self.logger = logging.getLogger(__name__)

        # basic DQN parameters
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.training_start = training_start
        self.target_update_freq = target_update_freq

        # tf.Optimizer
        self.optimizer = optimizer

        # additional configuration options
        self.gradient_clipping = gradient_clipping
        self.render_training = render_training
        self.reward_clipping = reward_clipping

        # misc
        self.debug = debug
        self.tensorboard_dir = tensorboard_dir
        self.rolling_reward_mean = rolling_reward_mean
        self.clean_tensorboard_runs = clean_tensorboard_runs
        self.tensorboard_suffix = tensorboard_suffix

        # calculate timestep where epsilon reaches its final value
        self.schedule_timesteps = int(total_timesteps * exploration_fraction)

        # concat name of instance to path -> distinction between saved instances
        self.checkpoint_dir = '{}/{}/'.format(checkpoint_dir, name)

        # whether to use tensorboard or not
        self.use_tensorboard = use_tensorboard

        # env specific parameter
        self.obs_shape = env.observation_space.shape
        self.num_actions = env.action_space.n

        # tf scopes
        self.Q_SCOPE = 'q_network'
        self.TARGET_SCOPE = 'target_network'

        # init q network, target network, replay buffer
        self.q_net, self.q_in = build_network(self.obs_shape, self.num_actions, network_type=network_type,
                                              name=self.Q_SCOPE)
        self.target_net, self.target_in = build_network(self.obs_shape, self.num_actions, network_type=network_type,
                                                        name=self.TARGET_SCOPE)
        self.replay_buffer = ReplayBuffer(int(buffer_size))

        # save variables from q and target network for
        self.q_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Q_SCOPE)
        self.target_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.TARGET_SCOPE)

        # define additional variables used in loss function
        self._L_r = tf.placeholder(tf.float32, (None,), name='loss_rewards')
        self._L_a = tf.placeholder(tf.int32, (None,), name='loss_actions')
        self._L_d = tf.placeholder(tf.float32, (None,), name='loss_dones')

        # epsilon schedule
        self.eps = LinearSchedule(self.schedule_timesteps, final_p=final_eps)

        # init optimizer with loss to minimize
        self.opt = self.optimizer(self.lr)
        self.gradients = self.opt.compute_gradients(self._loss(), var_list=self.q_net_vars)

        # clip gradients
        if self.gradient_clipping is not None:
            for idx, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[idx] = (tf.clip_by_norm(grad, self.gradient_clipping), var)

        # create training op
        self.train_op = self.opt.apply_gradients(self.gradients)

        # update_target_fn will be called periodically to copy Q network to target Q network
        self.update_target_expr = []
        for var, var_target in zip(sorted(self.q_net_vars, key=lambda v: v.name),
                                   sorted(self.target_net_vars, key=lambda v: v.name)):
            self.update_target_expr.append(var_target.assign(var))
        self.update_target_expr = tf.group(*self.update_target_expr)

        # global tf.Session and Graph init
        self.sess = tf.Session()

        # launch debug session if requested
        if self.debug:
            self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "localhost:6064")

        # create tensorboard summaries
        if self.use_tensorboard:
            self._setup_tensorboard()

        # init variables etc.
        self.sess.run(tf.global_variables_initializer())

        # sync networks before training
        self.sess.run(self.update_target_expr)

        # flag indicating whether this instance is completely trained
        self.is_trained = False

        # if this instance is working with checkpoints, we'll check whether
        # one is already there. if so, we continue training from that checkpoint,
        # i.e. load the saved weights into target and online network.
        if self.use_checkpoints:

            # remove old weights if needed
            if self.clean_previous_weights:
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
            scalar_summary('QTa_{}'.format(a_i + 1), tf.reduce_mean(self.target_net[:, a_i]), scope='Q-Values')
            scalar_summary('Qa_{}'.format(a_i + 1), tf.reduce_mean(self.q_net[:, a_i]), scope='Q-Values')

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

            # calculate target
            y = self._L_r + (self.gamma * (1.0 - self._L_d) * tf.reduce_max(self.target_net, axis=1))

            # select q value of taken action
            qj = tf.reduce_sum(self.q_net * tf.one_hot(self._L_a, self.num_actions), 1)

            loss = tf.losses.huber_loss(y, qj)

        if self.use_tensorboard:
            scalar_summary('target', tf.reduce_mean(y))
            scalar_summary('huber-loss', tf.reduce_mean(loss))
            tf.summary.histogram('selected_Q', qj)

        return tf.reduce_mean(loss)

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
        obs = self.env.reset()

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
                action = np.argmax(self.sess.run(self.q_net, {self.q_in: [obs]}), axis=1)
                assert len(action) == 1, 'only one action can be taken!'
                action = action[0]

            # take action in environment
            new_obs, reward, done, _ = self.env.step(action)

            # clip reward
            if self.reward_clipping:
                reward = 1 if reward > 0 else -1 if reward < 0 else 0

            # store new transition
            self.replay_buffer.add(obs, action, reward, new_obs, float(done))

            # t + 1 -> t
            obs = new_obs

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
                obs = self.env.reset()

            if t >= self.training_start:

                # calculate rolling reward
                rr = np.mean(episode_rewards[-self.rolling_reward_mean:]) if len(episode_rewards) > 0 else 0.0

                # small episode results table
                if done:
                    result_table = [
                        ['t', t],
                        ['episode', len(episode_rewards)],
                        ['mean_reward [20]', np.mean(episode_rewards[-self.rolling_reward_mean:])],
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

                o, a, r, no, d = self.replay_buffer.sample(self.batch_size)

                # collect summaries if needed or just train
                if self.use_tensorboard:
                    summary, _ = self.sess.run([self.merge_op, self.train_op],
                                               feed_dict={self.target_in: no, self.q_in: o, self._L_r: r, self._L_a: a,
                                                          self._L_d: d, self.eps_ph: epsilon,
                                                          self.rew_ph: rr})
                    self.writer.add_summary(summary, t)
                else:
                    self.sess.run(self.train_op, feed_dict={self.target_in: no, self.q_in: o, self._L_r: r,
                                                            self._L_a: a, self._L_d: d})

                # sync target network every C steps
                if (t - self.training_start) % self.target_update_freq == 0:
                    self.sess.run(self.update_target_expr)

            if self.solved_callback is not None:
                if self.solved_callback(episode_rewards):
                    self.logger.info('Solved!')
                    break

        # total reward of last (possibly interrupted) episode
        episode_rewards.append(np.sum(episode_reward_series[-1]))

        # finalize training, e.g. set flags, write done-file
        self._finalize_training()

    def run(self):
        """ Runs policy on given environment """
        pass


if __name__ == '__main__':
    from forkan.rl import make
    env = make('CartPole-v0')

    agent = DQN(env,
                buffer_size=50000,
                total_timesteps=100000,
                training_start=1000,
                target_update_freq=500,
                exploration_fraction=0.1,
                lr=5e-4,
                gamma=1.,
                batch_size=32,
                clean_tensorboard_runs=True,
                )

    agent.learn()
