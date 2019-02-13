import os
import math
import logging
import numpy as np

from forkan.common.utils import folder_to_npz, folder_to_unlabeled_npz

logger = logging.getLogger(__name__)


def load_image_dataset(name, target_size=[240, 240, 3], test_set=0.08):

    logger.info('Loading {} ...'.format(name))

    prefix = os.environ['HOME'] + '/.forkan/datasets/'
    dataset_file = '{}/{}.npz'.format(prefix, name)

    # check whether we need to generate dataset archive
    if not os.path.isfile(dataset_file):
        folder_to_npz(prefix, name, target_size=target_size, test_set=test_set)

    dataset_zip = np.load(dataset_file, encoding='latin1')

    logger.info('Finished loading.')
    return (dataset_zip['x_train'], dataset_zip['y_train']), (dataset_zip['x_test'], dataset_zip['y_test']), \
           (dataset_zip['idx2label'], dataset_zip['label2idx'])


def load_unlabeled_image_dataset(name, test_set=None):

    logger.info('Loading {} ...'.format(name))

    atari = False

    if 'atari-' in name:
        atari = True
        name = name.replace('atari-', '')

    prefix = os.environ['HOME'] + '/.forkan/datasets/'
    dataset_file = '{}/{}.npz'.format(prefix, name)

    # check whether we need to generate dataset archive
    if not os.path.isfile(dataset_file):
        if atari:
            folder_to_unlabeled_npz(prefix, name, target_shape=[200, 160, 3])
        else:
            folder_to_unlabeled_npz(prefix, name)

    dataset_zip = np.load(dataset_file, encoding='latin1')
    x_train = dataset_zip['imgs']

    if test_set is not None:
        # split class into train & test images
        nb_test = math.floor(x_train.shape[0] * test_set)
        nb_train = x_train.shape[0] - nb_test

        # do the split
        train, test = x_train[:nb_train], x_train[nb_train:]
        logger.info('Splitting into {} train and {} test samples.'.format(nb_train, nb_test))
    else:
        train, test = x_train, None
        logger.info('Dataset has {} training samples.'.format(train.shape[0]))

    logger.info('Finished loading.')

    return train, test
