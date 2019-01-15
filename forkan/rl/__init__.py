import gym
import logging

from forkan.rl.base_agent import BaseAgent

from forkan.rl.env_wrapper import EnvWrapper
from forkan.rl.repeat_env import RepeatEnv
from forkan.rl.multi_env import MultiEnv
from forkan.rl.multi_stepper import MultiStepper

from forkan.rl.dqn.dqn import DQN
from forkan.rl.a2c.a2c import A2C


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

