import random
import numpy as np

from forkan.utils import store_args


class ReplayBuffer(object):

    @store_args
    def __init__(self, size):

        self._buffer = []
        self._maxsize = size
        self._idx = 0

    def __len__(self):
        return len(self._buffer)

    def add(self, obs, action, reward, new_obs, done):

        # pack sample in one tuple
        sample = (obs, action, reward, new_obs, done)

        # either append or overwrite
        if self._idx >= len(self):
            self._buffer.append(sample)
        else:
            self._buffer[self._idx] = sample

        # new index for next sample
        self._idx = int((self._idx + 1) % self._maxsize)

    def sample(self, batch_size):

        # sample BATCH_SIZE random numbers in correct range
        idxes = [random.randint(0, len(self) - 1) for _ in range(batch_size)]

        # init return arrays
        obse, acs, rs, nobse, dos = [], [], [], [], []

        for i in idxes:

            # restore and unpack data
            data = self._buffer[i]
            o, a, r, no, d = data

            # references are sufficient as the sample will only be used once for a gradient step
            obse.append(np.array(o, copy=False))
            acs.append(np.array(a, copy=False))
            rs.append(r)
            nobse.append(np.array(no, copy=False))
            dos.append(d)

        return np.array(obse), np.array(acs), np.array(rs), np.array(nobse), np.array(dos)
