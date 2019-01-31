import numpy as np
import tensorflow as tf

from forkan import EPS
from forkan.rl import BaseAgent, MultiStepper, MultiEnv
from forkan.common.policies import build_policy
from forkan.common.tf_utils import categorical_kl, value_by_index, get_trainable_variables, flat_grad, flat_concat, \
                                   assign_params_from_flat, scalar_summary, vector_summary

from tabulate import tabulate


class TRPO(BaseAgent):

    def __init__(self,
                 env,
                 alg_name='trpo',
                 name='default',
                 policy_type='pi-and-value',
                 total_timesteps=5e7,
                 vf_lr=1e-3,
                 gamma=0.99,
                 tmax=5,
                 delta=0.01,
                 backtracking_coef=0.8,
                 backtrack_steps=10,
                 cg_steps=10,
                 hess_damping_coeff=0.1,
                 v_train_iters=80,
                 max_grad_norm=None,
                 reward_clipping=None,
                 solved_callback=None,
                 print_freq=None,
                 render_training=False,
                 **kwargs,
                 ):
        """
        Implementation A2C aka Advantage Actor Critic introduced by Mnih.

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

        vf_lr: float
            learning rate for value function regression

        gamma: float
            discount factor gamma for bellman target

        tmax: int
            number of steps the agent acts on environment per timestep

        delta: float
           KL-constraint

        backtracking_coef: float
           backtracking coefficient during line search

        backtrack_steps: int
           maximum number of backtracking steps

        cg_steps: int
           maximum number of conjugate gradient iterations

        hess_damping_coeff: float
           for numerical stability when calculating the hessian vector product: Hv = (coef + I)H *v

        v_train_iters: int
            value function regression iterations per TRPO step

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

        # hyperparameters
        self.total_timesteps = total_timesteps
        self.policy_type = policy_type
        self.vf_lr = vf_lr
        self.gamma = gamma
        self.tmax = tmax

        # trpo specific parameters
        self.delta = delta
        self.backtracking_coef = backtracking_coef
        self.backtrack_steps = backtrack_steps
        self.cg_steps = cg_steps
        self.hess_damping_coeff = hess_damping_coeff
        self.v_train_iters = v_train_iters

        self.max_grad_norm = max_grad_norm
        self.reward_clipping = reward_clipping

        self.solved_callback = solved_callback
        self.print_freq = print_freq
        self.render_training = render_training

        super().__init__(env, alg_name, name, **kwargs)

        # number of environments
        if isinstance(env, MultiEnv):
            self.num_envs = env.num_envs
        else:
            self.num_envs = 1

        self.batch_size = self.num_envs * self.tmax

        # env specific parameter
        self.obs_shape = env.observation_space.shape
        self.num_actions = env.action_space.n

        # create value net, action net and policy output (and get input placeholder)
        self.obs_ph, self.logits, self.values, self.action = build_policy(self.obs_shape, self.num_actions, reuse=False,
                                                                          policy_type=self.policy_type)

        # relevant placeholders
        self.action_ph = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.d_rew_ph = tf.placeholder(tf.float32, shape=(None,), name='discounted-rewards')
        self.advantage_ph = tf.placeholder(tf.float32, shape=(None,), name='advantages')
        self.old_logits_ph = tf.placeholder(tf.float32, shape=(None, self.num_actions), name='old-logp')

        # store reference to theta
        self.theta_vars = get_trainable_variables('policy')

        # getter and setter for theta
        self._get_theta = lambda: self.sess.run(flat_concat(self.theta_vars))
        self._set_theta = lambda x: self.sess.run(assign_params_from_flat(x, self.theta_vars))

        """
        Part I: Policy Gradient
        
        Evaluate policy gradient on trajectories generated by old policy using Importance Sampling.    
        """

        # calculate log probabilities and log-prob ratio for IS
        logp = tf.nn.log_softmax(self.logits)
        old_logp = tf.nn.log_softmax(self.old_logits_ph)

        # select logp for action taken at t
        logp_t = value_by_index(logp, self.action_ph, self.num_actions)
        old_logp_t = value_by_index(old_logp, self.action_ph, self.num_actions)

        # IS weighing ratio
        log_ratio = tf.exp(logp_t - old_logp_t)

        # final policy gradient
        self.policy_loss = -tf.reduce_mean(log_ratio * self.advantage_ph)
        self.pg = flat_grad(self.policy_loss, self.theta_vars)

        """
        Part II: Hessian / FIM calculation
        """

        # calculate KL divergence and Hessian-Vector-Poduct as grad(grad-KL.T*x)
        self.kl = categorical_kl(old_logp, logp)
        grad_kl = flat_grad(self.kl, self.theta_vars)
        self.hvp_x_ph = tf.placeholder(tf.float32, shape=grad_kl.shape)
        Hx = flat_grad(tf.reduce_sum(grad_kl*self.hvp_x_ph), self.theta_vars)

        if self.hess_damping_coeff > 0:
            Hx += self.hess_damping_coeff * self.hvp_x_ph

        # finally, create function calculating hessian-vector-product; y is additional input dict
        self._Hx = lambda x, y: self.sess.run(Hx, {**y, self.hvp_x_ph: x})

        """
        Part III: Value Function Regression
        """

        # value function loss and training ops
        self.vf_loss = tf.reduce_mean((self.d_rew_ph - self.values)**2)
        self.train_vf_op = tf.train.AdamOptimizer(learning_rate=self.vf_lr).minimize(self.vf_loss)

        # multistepper that collects batches of experience
        self.multistepper = MultiStepper(self, self.env, self.tmax)

        # takes care of tensorboard, debug and checkpoint init
        self._finalize_init()

    def _setup_tensorboard(self):
        """
        Adds all variables that might help debugging to Tensorboard.
        At the end, the FileWriter is constructed pointing to the specified directory.

        """

        self.logger.info('Saving Tensorboard summaries to {}'.format(self.tensorboard_dir))

        self.ret_ph = tf.placeholder(tf.float32, (), name='mean-return')
        self.kl_ph = tf.placeholder(tf.float32, (), name='kl')
        self.pl_diff_ph = tf.placeholder(tf.float32, (), name='pl-diff')

        scalar_summary('mean-return', self.ret_ph)
        scalar_summary('kl', self.kl_ph)
        scalar_summary('pl-diff', self.pl_diff_ph)

        with tf.variable_scope('loss'):
            scalar_summary('value-loss', self.vf_loss)
            scalar_summary('policy-loss', self.policy_loss)
            # scalar_summary('policy-entropy', self.pi_entropy)

        with tf.variable_scope('value'):
            scalar_summary('value_target', tf.reduce_mean(self.d_rew_ph))
            scalar_summary('value', tf.reduce_mean(self.values))

        # plot network weights
        with tf.variable_scope('weights'):
            for pv in self.theta_vars: tf.summary.histogram('{}'.format(pv.name), pv)

        # gradient histograms
        with tf.variable_scope('gradients'):
            vector_summary('policy-gradient', self.pg)

    def step(self, obs):
        """ Returns logits, values and chosen actions given obs. """
        return self.sess.run([self.logits, self.values, self.action], feed_dict={
                self.obs_ph: obs
            })

    def cg(self, g, feed_d):
        """ Approximates x=H^-1*g using self._Hx as function for the hessian-vector-product. """
        x = np.zeros(shape=g.shape, dtype=np.float32)
        r = g.copy()
        p = r.copy()
        r_dot_old = np.dot(r, r)

        for _ in range(self.cg_steps):
            z = self._Hx(p, feed_d)
            alpha = r_dot_old/((np.dot(p, z)) + EPS)

            x = x + alpha*p
            r = r - alpha*z
            r_dot_new = np.dot(r, r)

            p = r + (r_dot_new/r_dot_old)*p
            r_dot_old = r_dot_new

        return x

    def learn(self):
        """ Trains Actor and Critic while satisfying the KL-constrained. """

        # reset env, store first observation
        self.multistepper.on_training_start()

        # real timesteps
        T = 0

        # some statistics
        cur_eps_rets = [0]*self.num_envs
        past_returns = []
        mean_ret = 0.0
        best_mean_ret = 0.0

        self.logger.info('Starting training!')

        pl_diff = 0.0

        while T < self.total_timesteps:

            # execute and collect trajectories
            batch_obs, batch_actions, batch_rewards, batch_dones, \
            batch_logits, batch_values, raw_rewards = self.multistepper.step()

            T += self.tmax

            # we check for each env whether it finised, otherwise store rewards
            # resets are not needed, they happen in the threads after episode termination
            for n, (r_t, d_t) in enumerate(zip(raw_rewards, batch_dones)):
                for (re, do) in zip(r_t, d_t):

                    # accumulate reward for this step
                    cur_eps_rets[n] += re

                    if do:
                        # store final reward in list
                        past_returns.append(cur_eps_rets[n])

                        # zero reward
                        cur_eps_rets[n] *= 0

                        # calculate mean returns and save model weights
                        if len(past_returns) > 20:
                            mean_ret = np.mean(past_returns[-20:])
                            if mean_ret > best_mean_ret:
                                best_mean_ret = mean_ret
                                self.logger.info('Storing weights with score {}'.format(best_mean_ret))
                                self._save('best')

                        if self.print_freq is not None and len(past_returns) % self.print_freq == 0:
                            result_table = [
                                ['T', T],
                                ['episode', len(past_returns)],
                                ['mean return', mean_ret],
                            ]

                            print('\n{}'.format(tabulate(result_table)))

            # estimate advantage values
            advantages = (batch_rewards - np.squeeze(batch_values))
            # TODO normalize advantages?

            food = {
                self.obs_ph: batch_obs,
                self.action_ph: batch_actions,
                self.d_rew_ph: batch_rewards,
                self.old_logits_ph: batch_logits,
                self.advantage_ph: advantages,
                self.ret_ph: mean_ret,
            }

            # estimate policy gradient
            g_hat, policy_loss_old = np.squeeze(self.sess.run([self.pg, self.policy_loss], feed_dict=food))

            # calculate search direction for line search
            s = self.cg(g_hat, food)

            # maximum step length
            beta = np.sqrt(2*self.delta/(np.dot(s, self._Hx(s, food))+EPS))

            # store old policy parameter
            theta_old = self._get_theta()

            """
            start linesearch in s, exponentially shrinking beta until KL contraint is satisfied (or not => reject)
            """
            # todo entropy as second check
            for j in range(self.backtrack_steps):

                self._set_theta(theta_old - self.backtracking_coef**j * s * beta)
                kl, policy_loss_new = self.sess.run([self.kl, self.policy_loss], feed_dict=food)

                if kl <= self.delta and policy_loss_new <= policy_loss_old:
                    # self.logger.info('Linesearch succeeded in step {}!'.format(j))
                    pl_diff = policy_loss_old - policy_loss_new
                    break
                elif j == self.backtrack_steps-1:
                    self.logger.info('Linesearch failed...'.format(j))
                    # reset policy to old parameters
                    self._set_theta(theta_old)

            # fit value function todo either use hyperparameter for or once per step
            for _ in range(1):
                self.sess.run(self.train_vf_op, feed_dict=food)

            # run training step using already computed data
            if self.use_tensorboard:
                food[self.kl_ph] = kl
                food[self.pl_diff_ph] = pl_diff
                sum = self.sess.run(self.merge_op, feed_dict=food)
                self.writer.add_summary(sum, T)

        # finalize training, e.g. set flags, write done-file
        self._finalize_training()

    def run(self, render=True):
        """ Runs policy on given environment """

        if not self.is_trained:
            self.logger.warning('Trying to run untrained model!')

        # set necessary parameters to their defaults
        reward = 0.0
        obs = self.env.reset()

        while True:

            _, _, action = self.step([obs])

            # act on environment with chosen action
            obs, rew, done, _ = self.env.step(action[0])
            reward += rew

            if render:
                self.env.render()

            if done:
                self.logger.info('Done! Reward {}'.format(reward))
                reward = 0.0
                obs = self.env.reset()
