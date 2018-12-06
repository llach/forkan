import gym
import sys
import logging

from forkan.rl.dqn.dqn import DQN
from forkan.rl.env_wrapper import EnvWrapper

algorithm_list = [
    'dqn',
]

logger = logging.getLogger(__name__)


def make(name, **kwargs):
    """ Makes gym env and wraps it in EnvWrapper """
    e = gym.make(name)
    return EnvWrapper(e, **kwargs)


def load_algorithm(alg_type, env_type, alg_kwargs={}, env_kwargs={}, preprocessor=None):
    logger.debug('Loading RL algorithm {} with environment {} ... '.format(alg_type, env_type))

    if preprocessor is not None:
        env_kwargs['preprocessor'] = preprocessor

    env = make(env_type, **env_kwargs)

    if alg_type == 'dqn':
        alg = DQN(env, **alg_kwargs)
    else:
        logger.error('RL algorithm {} is unknown!'.format(alg_type))
        sys.exit(1)

    return alg
