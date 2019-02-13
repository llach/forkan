import numpy as np

from forkan.common.utils import discount_with_dones


class MultiStepper(object):

    def __init__(self, model, env, tmax):
        """

        Acts tmax times on env using actions chosen by model.

        Parameters
        ----------
        model: BaseAgent
            needs to implement step() returning logits, state values and actions
        """

        self.env = env
        self.tmax = tmax

        # store model and important parameter
        self.model = model
        self.gamma = model.gamma
        self.batch_size = model.batch_size
        self.obs_shape = model.obs_shape
        self.num_actions = model.num_actions
        self.num_envs = model.num_envs

        self.obs_t = None
        self.d_t = None

    def on_training_start(self):
        """ Initialized variables before training starts. """
        self.obs_t = self.env.reset()
        self.d_t = [False] * self.num_envs

    def step(self):

        # We initialize the lists that will contain the batch of experiences
        batch_obs, batch_rewards, batch_actions, batch_values, batch_dones, batch_logits = [], [], [], [], [], []

        # perform multisteps
        for n in range(self.tmax):

            # calculate pi & state value
            logits, values, actions = self.model.step(self.obs_t)

            # observe o+1, r+1, d+1
            obs_tp1, r_t, d_tp1, _ = self.env.step(actions)

            # Append the experiences
            batch_obs.append(np.copy(self.obs_t))
            batch_actions.append(actions)
            batch_logits.append(logits)
            batch_values.append(values)
            batch_dones.append(self.d_t)
            batch_rewards.append(r_t)

            # t <-- t + 1
            self.obs_t = obs_tp1
            self.d_t = d_tp1

        # we append the last dones and cut away the first ones later to align dones with obs and rewards
        batch_dones.append(self.d_t)

        # Batch of steps to batch of rollouts
        batch_obs = np.asarray(batch_obs, dtype=np.float32).swapaxes(1, 0).reshape((self.batch_size,) + self.obs_shape)
        batch_rewards = np.asarray(batch_rewards, dtype=np.float32).swapaxes(1, 0)
        batch_actions = np.asarray(batch_actions, dtype=np.int32).swapaxes(1, 0)
        batch_values = np.asarray(batch_values, dtype=np.float32).swapaxes(1, 0)
        batch_logits = np.asarray(batch_logits, dtype=np.float32).swapaxes(1, 0)
        batch_dones = np.asarray(batch_dones, dtype=np.bool).swapaxes(1, 0)
        batch_dones = batch_dones[:, 1:]

        # we need not discounted or otherwise manipulated rewards in the model for correct reward calculation
        raw_rewards = batch_rewards.copy()

        if self.gamma > 0.0:
            # state value
            _, last_values, _ = self.model.step(self.obs_t)

            for n, (rewards, dones, value) in enumerate(zip(batch_rewards, batch_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                batch_rewards[n] = rewards

        batch_actions = batch_actions.reshape((self.batch_size,))
        batch_logits = batch_logits.reshape((self.batch_size, self.num_actions))

        batch_rewards = batch_rewards.flatten()
        batch_values = np.expand_dims(batch_values.flatten(), -1)

        return batch_obs, batch_actions, batch_rewards, batch_dones, batch_logits, batch_values, raw_rewards

