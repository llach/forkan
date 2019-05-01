import numpy as np

from forkan import dataset_path
from forkan.datasets.dsprites import load_dsprites, load_dsprites_one_fixed
from forkan.datasets.image import load_unlabeled_image_dataset
from forkan.datasets.mnist import load_mnist


def load_set(name):
    return np.load('{}/{}.npz'.format(dataset_path, name))['data']


def load_atari_normalized(env):
    name = env.replace('NoFrameskip', '').lower().split('-')[0]
    return np.load('{}/{}-normalized.npz'.format(dataset_path, name))['data']


def load_pendulum():
    return np.load('{}/pendulum-visual-random-normalized-cut.npz'.format(dataset_path))['data']


def load_uniform_pendulum():
    return np.load('{}/pendulum-visual-uniform.npz'.format(dataset_path))['data']
