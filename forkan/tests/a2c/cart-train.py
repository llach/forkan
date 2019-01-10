import argparse

from forkan.rl import make, A2C


# mark env as solved once rewards were higher than 195 50 episodes in a row
def solved_callback(rewards):
    if len(rewards) < 50:
        return False

    for r in rewards[-50:]:
        if r < 195:
            return False

    return True


# algorithm parameters
a2c_conf = {
    'name': 'cart-a2c',
    'total_timesteps': 1e5,
    'lr': 5e-4,
    'gamma': .95,
    'entropy_coef': 0.01,
    'v_loss_coef': 0.2,
    'gradient_clipping': None,
    'reward_clipping': None,
    'use_tensorboard': True,
    'clean_tensorboard_runs': True,
    'clean_previous_weights': True,
    'print_freq': 20,
    'solved_callback': solved_callback,
}

# environment parameters
env_conf = {
    'eid': 'CartPole-v0',
    'num_envs': 4,
}

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--run', '-r', action='store_true')
args = parser.parse_args()

# remove keys from config so that the correct environment will be created
if args.run:
    env_conf.pop('num_envs')
    a2c_conf['clean_previous_weights'] = False

e = make(**env_conf)

alg = A2C(e, **a2c_conf)

if args.run:
    print('Running a2c on {}'.format(env_conf['eid']))
    alg.run()
else:
    print('Learning with a2c on {}'.format(env_conf['eid']))
    alg.learn()
