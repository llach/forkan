import gym

from forkan.rl import DQN
from forkan.rl.atari_env import AtariEnv
from forkan.rl.repeat_env import RepeatEnv

dqn_conf = {
    'name': 'breakout-dqn',
    'network_type': 'nature-cnn',
    'buffer_size': 1e6,
    'total_timesteps': 5e7,
    'training_start': 5e4,
    'target_update_freq': 1e4,
    'exploration_fraction': 0.1,
    'gamma': .99,
    'batch_size': 32,
    'prioritized_replay': True,
    'double_q': True,
    'dueling': True,
    'clean_tensorboard_runs': True,
    'clean_previous_weights': True,
}

atari_env_conf = {
    'target_shape': (200, 160),
    'grayscale': True,
}

wrapper_conf = {
    'observation_buffer_size': 4,
    'action_repetition': 4,
}

e = gym.make('Breakout-v0')
e = AtariEnv(e, **atari_env_conf)
e = RepeatEnv(e, **wrapper_conf)

alg = DQN(e, **dqn_conf)
alg.learn()

