import numpy as np

from forkan import dataset_path

from forkan.datasets.mnist import load_mnist
from forkan.datasets.image import load_unlabeled_image_dataset
from forkan.datasets.dsprites import load_dsprites, load_dsprites_one_fixed

def load_atari_normalized(env):
    name = env.replace('NoFrameskip', '').lower().split('-')[0]
    return np.load('{}/{}-normalized.npz'.format(dataset_path, name))['data']