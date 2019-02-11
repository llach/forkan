import numpy as np

from collections import deque
from gym import Space


class RepeatEnv(object):

    def __init__(self, 
                 env,
                 action_repetition=1,
                 observation_buffer_size=1,
                 preprocessor=None,
                 flatten_observations=False,
                 buffer_last=True,
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

        flatten_observations: bool
            if True, will call obs.flatten(), otherwise observation buffer is
            returned with batch-style shape: (BUFFER_SIZE, OBS_SHAPE)

        buffer_last: bool
            if True, obs will have shape (OBS_SHAPE, BUFFER_SIZE), and (BUFFER_SIZE, OBS_SHAPE) otherwise
        """

        self.env = env
        self.action_repetition = action_repetition
        self.observation_buffer_size = observation_buffer_size
        self.flatten_observations = flatten_observations
        self.buffer_last = buffer_last
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
            self.oshape = (1, self.preprocessor.latent_dim)
        else:
            self.oshape = self.env.observation_space.shape

        # store for empty buffer init
        self.single_observation_shape = self.oshape

        # if observation_buffer_size == 1, we leave out the first (batch) dimension of size 1.
        # otherwise we'd break compatibility with MLPs
        if self.flatten_observations:
            self.observation_space = Space((self.observation_buffer_size * np.product(self.oshape),), np.int8)
        else:
            if self.buffer_last:
                self.observation_space = Space(self.oshape + (self.observation_buffer_size,), np.int8)
            else:
                self.observation_space = Space((self.observation_buffer_size,) + self.oshape, np.int8)

        # quick pointer to observation space shape
        self.oshape = self.observation_space.shape

        # at the beginning, we want only zeros in buffer
        self._empty_observation_buffer()

    def _empty_observation_buffer(self):
        """ Fills buffer with observation space sized zero-arrays """
        for _ in range(self.observation_buffer_size):
            self.obs_buffer.append(np.zeros(self.single_observation_shape))

    def _transform_obs(self, obs):
        """ Processes observations based on Wrapper config """

        # pass to preprocessor
        if self.preprocessor is not None:
            obs = self.preprocessor.process(np.expand_dims(obs, axis=0))
        return obs

    def _transform_buffer(self):
        """ Applies configures transformations to observation buffer """

        ol = np.array(self.obs_buffer)

        if self.buffer_last:
            ol = np.moveaxis(ol, 0, len(ol.shape)-1)

        # sqeeze away dimension of size 1
        ol = np.squeeze(ol)

        if self.flatten_observations:
            ol = ol.flatten()

        return ol

    def step(self, action):
        """ Executes action on env, repeated M times """

        for _ in range(self.action_repetition):
            obs, reward, done, info = self.env.step(action)
            obs = self._transform_obs(obs)

            self.obs_buffer.append(obs)

        return self._transform_buffer(), reward, done, info

    def render(self):
        """ Renders environment """
        return self.env.render()

    def reset(self):
        """ Resets env and observation buffer """
        self._empty_observation_buffer()
        obs = self.env.reset()
        obs = self._transform_obs(obs)

        self.obs_buffer.append(obs)

        return self._transform_buffer()
