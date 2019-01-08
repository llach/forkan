from forkan.rl import make, A2C


def solved_callback(rewards):
    if len(rewards) < 50:
        return False

    for r in rewards[-50:]:
        if r < 195:
            return False

    return True


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

env_conf = {
    'eid': 'CartPole-v0',
    'num_envs': 4,
}

e = make(**env_conf)

alg = A2C(e, **a2c_conf)
alg.learn()
