import numpy as np

from PIL import Image


class AtariEnv(object):

    def __init__(self,
                 env,
                 target_shape=(200, 160),
                 grayscale=False,
                 ):
        """
        Environment resizes ALE outputs and grayscales if needed.

        env: gym.Environment
            OpenAI Gym Environment that is to be wrapped

        target_shape: tuple
            input shape for preprocessor

        grayscale: bool
            whether to convert image to grayscale
        """

        self.env = env
        self.target_shape = target_shape
        self.grayscale = grayscale

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        #
        if self.grayscale:
            self.observation_space.shape = self.target_shape
        else:
            self.observation_space.shape = self.target_shape + (self.observation_space.shape[-1],)

    def _transform_obs(self, obs):
        """ Processes observations based on Wrapper config """

        img = Image.fromarray(obs)

        # resize image
        if self.target_shape is not None:
            img = img.resize((self.target_shape[1], self.target_shape[0]))

        # convert to grayscale
        if self.grayscale:
            img = img.convert('L')

        return np.array(img)

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        obs = self._transform_obs(obs)

        return obs, reward, done, info

    def render(self):
        """ Renders environment """
        return self.env.render()

    def reset(self):
        obs = self.env.reset()
        obs = self._transform_obs(obs)

        return obs
