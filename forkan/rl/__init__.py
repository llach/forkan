import gym
import inspect
import logging

from forkan.rl.env_wrapper import EnvWrapper
from forkan.rl.envs import AtariPrep, MultiEnv, LazyVAE, FakeLazyVAE, PendulumRenderEnv,\
    PendulumRenderVAEEnv, PendulumVAEStackEnv, VecVAEStack

from forkan.rl.envs.multi_stepper import MultiStepper

from forkan.rl.base_agent import BaseAgent
from forkan.rl.algos import DQN, A2C, TRPO


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

        # wrap env if constructor arguments are found in kwargs
        for env in [AtariPrep]:
            if any(map(lambda x: x in kwargs.keys(), get_filtered_args(env))):
                e = env(e, **kwargs)

        return e

    # either we thread the env or return the constructed environment
    if 'num_envs' in kwargs:
        return MultiEnv(kwargs['num_envs'], maker)
    else:
        return maker()
