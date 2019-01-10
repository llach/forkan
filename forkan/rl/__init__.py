import gym
import sys
import logging

from forkan.rl.base_agent import BaseAgent

from forkan.rl.env_wrapper import EnvWrapper
from forkan.rl.repeat_env import RepeatEnv
from forkan.rl.multi_env import MultiEnv
from forkan.rl.multi_stepper import MultiStepper

from forkan.rl.dqn.dqn import DQN
from forkan.rl.a2c.a2c import A2C

algorithm_list = [
    'dqn',
]

logger = logging.getLogger(__name__)


def make(eid,
         num_envs=None
         ):
    """ Makes gym env and wraps it in EnvWrapper """

    def maker():
        e = gym.make(eid)
        return e

    # either we thread the env or return the constructed environment
    if num_envs is not None:
        return MultiEnv(num_envs, maker)
    else:
        return maker()


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
