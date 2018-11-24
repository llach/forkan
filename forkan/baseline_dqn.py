import tensorflow as tf
import tensorflow.contrib.layers as layers
from tabulate import tabulate

import numpy as np
import random
import gym

from forkan.rl.dqn.replay_buffer import ReplayBuffer
from forkan.schedules import LinearSchedule

lr = 5e-4
total_timesteps = 100000
buffer_size = 50000
exploration_fraction = 0.1
exploration_final_eps = 0.02
train_freq = 1
batch_size = 32
print_freq = 5
learning_starts = 1000
gamma = 1.0
target_network_update_freq = 500


def build_q_func(input, name):
    with tf.variable_scope(name):
        l1 = layers.fully_connected(input, num_outputs=256, activation_fn=None)
        a1 = tf.nn.relu(l1)
        out = layers.fully_connected(a1, num_outputs=num_actions, activation_fn=None)
    return out


env = gym.make('CartPole-v0')


Q_SCOPE = 'online'
T_SCOPE = 'target'

num_actions=env.action_space.n
obs_shape = (None, ) + env.observation_space.shape
optimizer = tf.train.AdamOptimizer(learning_rate=lr)

# set up placeholders
obs_t_input = tf.placeholder(tf.float32, obs_shape, name='obs_t')
act_t_ph = tf.placeholder(tf.int32, [None], name="action")
rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
obs_tp1_input = tf.placeholder(tf.float32, obs_shape, name='obs_tp1')
done_mask_ph = tf.placeholder(tf.float32, [None], name="done")

# q network evaluation
q_t = build_q_func(obs_t_input, Q_SCOPE)
q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=Q_SCOPE)

# target q network evalution
q_tp1 = build_q_func(obs_tp1_input, T_SCOPE)
target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=T_SCOPE)

# q scores for actions which we know were selected in the given state.
q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

# compute estimate of best possible value starting from state at t + 1
q_tp1_best = tf.reduce_max(q_tp1, 1)
q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

# compute RHS of bellman equation
q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

# compute the error (potentially clipped)

td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
errors = tf.losses.huber_loss(tf.stop_gradient(q_t_selected_target), q_t_selected)
e2 = tf.Print(errors, [errors])
weighted_error = tf.reduce_mean(e2)

sess = tf.Session()
# compute optimization op (potentially with gradient clipping)
optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

# update_target_fn will be called periodically to copy Q network to target Q network
update_target_expr = []
for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                           sorted(target_q_func_vars, key=lambda v: v.name)):
    update_target_expr.append(var_target.assign(var))
update_target_expr = tf.group(*update_target_expr)


def update_target():
    sess.run(update_target_expr)


def act(eps, obs):
    epsilon = eps
    _rand = np.random.choice([True, False], p=[epsilon, 1 - epsilon])
    if _rand:
        action = env.action_space.sample()
    else:
        action = np.argmax(sess.run(q_t, {obs_t_input: [obs]}), axis=1)
        assert len(action) == 1, 'only one action can be taken!'
        action = action[0]

    return action


def learn():

    replay_buffer = ReplayBuffer(buffer_size)
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(max_t=int(exploration_fraction * total_timesteps),
                                 final=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    sess.run(tf.global_variables_initializer())
    update_target()

    episode_rewards = [0.0]
    obs = env.reset()

    for t in range(total_timesteps):

        # Take action and update exploration to the newest value
        update_eps = exploration.value(t)
        action = act(update_eps, obs)

        env_action = action
        new_obs, rew, done, _ = env.step(env_action)

        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs

        episode_rewards[-1] += rew
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

        if t == learning_starts:
            print('learning starts')

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            _ = sess.run([optimize_expr], feed_dict={obs_t_input: obses_t,
                                                   act_t_ph: actions,
                                                   rew_t_ph: rewards,
                                                   obs_tp1_input: obses_tp1,
                                                   done_mask_ph: dones})

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target()

        if done and len(episode_rewards) % print_freq == 0:
            result_table = [
                ['tt', t],
                ['episode', len(episode_rewards)],
                ['mean reward', np.mean(episode_rewards[-20:])],
                ['epsilon', update_eps]
            ]
            print('\n{}'.format(tabulate(result_table)))


learn()
