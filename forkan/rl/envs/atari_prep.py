import logging
import numpy as np


from gym import Space
from forkan.rl import EnvWrapper
from collections import deque
from skimage.transform import resize


class AtariPrep(EnvWrapper):

    def __init__(self,
                 env,
                 num_frames=4,
                 buffer_last=True,
                 target_shape=None,
                 crop_ranges=None,
                 **kwargs,
                 ):
        """
        This class takes care of transforming atari observations. In particular, it allows to buffer the last N
        observations as grayscale images in place of RGB channels.

        env: gym.Environment
            OpenAI Gym Environment that is to be wrapped

        observation_buffer_size: int
            number of last observations that are returned by step()

        buffer_last: bool
            if True, obs will have shape (OBS_SHAPE, BUFFER_SIZE), and (BUFFER_SIZE, OBS_SHAPE) otherwise

        target_shape: tuple
            image shape after resizing

        crop_ranges: list of two tuples
            crops axes according to passed tuples
        """

        self.logger = logging.getLogger(__name__)

        # inheriting from EnvWrapper and passing it an env makes spaces available.
        super().__init__(env)

        self.cr = crop_ranges
        self.target_shape = target_shape
        self.num_frames = num_frames
        self.buffer_last = buffer_last

        # sanity checks
        assert self.num_frames > 0

        if self.num_frames < 2:
            self.logger.warning('Using only one grayscale frame in observation buffer!')

        # initialize observation buffer
        self.obs_buffer = deque(maxlen=self.num_frames)

        self.did = False

        # frames can be resized or cropped
        self.image_shape = self.observation_space.shape[:-1] if target_shape is None else target_shape
        self.image_shape = self.image_shape if self.cr is None else (self.cr[0][1]-self.cr[0][0],
                                                                     self.cr[1][1]-self.cr[1][0])

        # analog to channels_first and channels_last
        if self.buffer_last:
            self.observation_space = Space(self.image_shape + (self.num_frames,), np.int8)
        else:
            self.observation_space = Space((self.num_frames,) + self.image_shape, np.int8)

        # at the beginning, we want only zeros in buffer
        self._empty_observation_buffer()

    def _empty_observation_buffer(self):
        """ Fills buffer with observation space sized zero-arrays. """
        for _ in range(self.num_frames):
            if self.buffer_last:
                self.obs_buffer.append(np.zeros(self.observation_space.shape[:-1]))
            else:
                self.obs_buffer.append(np.zeros(self.observation_space.shape[1:]))

    def _transform_obs(self, o):
        """ Transforms given observation o to grayscale. """

        # different sets of weights:
        # [0.299, 0.587, 0.114]
        # [0.2125, 0.7154, 0.0721]
        o_trans = np.dot(o[..., :3], [0.299, 0.587, 0.114])

        # resize
        if self.target_shape is not None and self.target_shape != o_trans.shape:
            o_trans = resize(o_trans, self.target_shape, anti_aliasing=True, mode='reflect')

        # crop
        if self.cr is not None:
            o_trans = o_trans[self.cr[0][0]:self.cr[0][1], self.cr[1][0]:self.cr[1][1]]

        return o_trans

    def _transform_buffer(self):
        """ Applies configures transformations to observation buffer. """

        ol = np.array(self.obs_buffer)

        if self.buffer_last:
            ol = np.moveaxis(ol, 0, len(ol.shape)-1)

        return ol

    def step(self, action):
        """ Execute action and transform observations. """

        obs, reward, done, info = self.env.step(action)
        obs = self._transform_obs(obs)

        self.obs_buffer.append(obs)

        return self._transform_buffer(), reward, done, info

    def reset(self):
        """ Resets env and observation buffer. """
        self._empty_observation_buffer()
        self.obs_buffer.append(self._transform_obs(self.env.reset()))

        return self._transform_buffer()

    def render(self, mode='human'):

        # render normally
        if mode != 'rgb_array':
            return self.env.render()
        else: # return last, transformed frame
            return self.obs_buffer[-1]

