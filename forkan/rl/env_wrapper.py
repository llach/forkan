import numpy as np

from collections import deque
from gym.core import Space
from PIL import Image

class EnvWrapper(object):

    def __init__(self, 
                 env,
                 action_repetition=1,
                 observation_buffer_size=1,
                 preprocessor=None,
                 target_shape=None,
                 ):
        """
        This class is used to wrap gym environments in order to add
        additional features:

        - a list of the last N observation can be returned form step(action) rather than
          only one observation

        - actions passed to step(action) may be repeated M times


        env: gym.Environment
            OpenAI Gym Environment that is to be wrapped

        action_repetition: int
            number of repetitions of an action passed to step()

        observation_buffer_size: int
            number of last observations that are returned by step()

        preprocessor: Object
            object with a process() method that processes raw
            environment observations, e.g. a VAE

        target_shape: tuple
            input shape for preprocessor
        """

        self.env = env
        self.action_repetition = action_repetition
        self.target_shape = target_shape
        self.observation_buffer_size = observation_buffer_size
        self.preprocessor = preprocessor

        # we must execute actions and return at least one observation
        assert self.action_repetition > 0
        assert self.observation_buffer_size > 0

        # initialize observation buffer
        self.obs_buffer = deque(maxlen=self.observation_buffer_size)

        # we set properties of the environment as our own,
        # so the wrapper exposes all information needed to construct model
        self.action_space = self.env.action_space

        # if output is transformed by a preprocessor, adjust the observation space shape
        if self.preprocessor is not None:
            self.oshape = (self.preprocessor.latent_dim, )
        else:
            self.oshape = self.env.observation_space.shape

        # if observation_buffer_size == 1, we leave out the dimension of size 1.
        # otherwise we'd break compatibility with MLPs
        if observation_buffer_size == 1:
            self.observation_space = Space(self.oshape, np.int8)
        else:
            self.observation_space = Space(((self.observation_buffer_size,) + self.oshape), np.int8)

        # at the beginning, we want only zeros in buffer
        self._empty_observation_buffer()

    def _empty_observation_buffer(self):
        """ Fills buffer with observation space sized zero-arrays """
        for _ in range(self.observation_buffer_size):
            self.obs_buffer.append(np.zeros(self.oshape))

    def _transform_obs(self, obs):
        """ Processes observations based on Wrapper config """

        # pass to preprocessor
        if self.preprocessor is not None:
            if self.target_shape is not None:
                obs = np.array(Image.fromarray(obs).resize((self.target_shape[1], self.target_shape[0])), dtype=np.float32)
            obs = self.preprocessor.process(np.expand_dims(obs, axis=0))

        return obs

    def step(self, action):
        """ Executes action M times on env """

        for _ in range(self.action_repetition):
            obs, reward, done, info = self.env.step(action)
            obs = self._transform_obs(obs)

            self.obs_buffer.append(obs)

        obs_list = list(self.obs_buffer)
        if self.observation_buffer_size == 1:
            obs_list = np.reshape(obs_list, self.oshape)

        return obs_list, reward, done, info

    def render(self):
        """ Renders environment """
        return self.env.render()

    def reset(self):
        """ Resets env and observation buffer """
        self._empty_observation_buffer()
        obs = self.env.reset()
        obs = self._transform_obs(obs)

        self.obs_buffer.append(obs)

        obs_list = list(self.obs_buffer)
        if self.observation_buffer_size == 1:
            obs_list = np.reshape(obs_list, self.oshape)

        return obs_list
