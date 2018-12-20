import gym

from forkan.rl import A2C
from forkan.rl.env_wrapper import EnvWrapper


def solved_callback(rewards):
    if len(rewards) < 50:
        return False

    for r in rewards[-50:]:
        if r < 195:
            return False

    return True


a2c_conf = {
    'name': 'cart-a2c',
    'total_timesteps': 5e4,
    'lr': 5e-3,
    'gamma': .99,
    'beta': 0.02,
    'gradient_clipping': None,
    'reward_clipping': 1,
    'use_tensorboard': True,
    'clean_tensorboard_runs': True,
    'clean_previous_weights': True,
    'solved_callback': solved_callback,
}

e = gym.make('CartPole-v0')
e = EnvWrapper(e)

alg = A2C(e, **a2c_conf)
alg.learn()
