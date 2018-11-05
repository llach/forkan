import os
import logging
import numpy as np

from forkan.utils import folder_to_npz

logger = logging.getLogger(__name__)

def load_image_dataset(name, target_size=[240, 240, 3], test_set=0.08):

    logger.info('Loading {} ...'.format(name))

    prefix = os.environ['HOME'] + '/.keras/datasets/'
    dataset_file = '{}/{}.npz'.format(prefix, name)

    # check whether we need to generate dataset archive
    if not os.path.isfile(dataset_file):
        folder_to_npz(prefix, name, target_size=target_size, test_set=test_set)

    zip = np.load(dataset_file, encoding='latin1')

    logger.info('Finished loading.')
    return (zip['x_train'], zip['y_train']), (zip['x_test'], zip['y_test']), (zip['idx2label'], zip['label2idx'])

