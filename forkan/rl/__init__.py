import gym
import inspect
import logging

from forkan.rl.env_wrapper import EnvWrapper
from forkan.rl.envs import VecVAEStack, VAEStack, VAEGradient

logger = logging.getLogger(__name__)


def make(**kwargs):
    """ Makes gym env and wraps it in EnvWrapper """

    # returns constructor args while ignoring certain names
    get_filtered_args = lambda y: list(
        filter(lambda x: x not in ['self', 'env'], inspect.getfullargspec(y.__init__).args))

    def maker():
        # filter kwargs for gym args
        gym_args = {k: v for (k, v) in kwargs.items() if k in ['id', 'frameskip', 'game', 'obs_type']}

        e = gym.make(**gym_args)

        return e


    return maker()
