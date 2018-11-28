import gym

from forkan.rl.dqn.dqn import DQN
from forkan.rl.env_wrapper import EnvWrapper


def make(name, **kwargs):
    """ Makes gym env and wraps it in EnvWrapper """
    e = gym.make(name)
    return EnvWrapper(e, **kwargs)
