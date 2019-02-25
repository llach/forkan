import logging

from gym import spaces
from forkan.rl import EnvWrapper


class FakeLazyVAE(EnvWrapper):

    def __init__(self,
                 env,
                 latents=20,
                 ):
        """
        Wraps some Atari Env (FrameStack, VecEnv ...) and adjusts

        env: gym.Environment
            FrameStackEnv that buffers LazyFrames
        """

        self.logger = logging.getLogger(__name__)

        # inheriting from EnvWrapper makes all attributes of self.env available
        super().__init__(env)

        bs = self.observation_space.shape[-1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(bs*2*latents,))

