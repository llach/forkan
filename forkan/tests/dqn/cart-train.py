import gym

from forkan.rl import DQN
from forkan.rl.repeat_env import RepeatEnv


def solved_callback(rewards):
    if len(rewards) < 50:
        return False

    for r in rewards[-50:]:
        if r < 195:
            return False

    return True


dqn_conf = {
    'name': 'cart-dqn',
    'buffer_size': 5e4,
    'total_timesteps': 1e5,
    'training_start': 1e3,
    'target_update_freq': 500,
    'exploration_fraction': 0.1,
    'prioritized_replay': True,
    'double_q': True,
    'dueling': True,
    'lr': 5e-4,
    'gamma': 1.,
    'batch_size': 128,
    'gradient_clipping': 10,
    'reward_clipping': 1,
    'clean_tensorboard_runs': True,
    'clean_previous_weights': True,
    'solved_callback': solved_callback,
}

e = gym.make('CartPole-v0')
e = RepeatEnv(e)

alg = DQN(e, **dqn_conf)
alg.learn()
