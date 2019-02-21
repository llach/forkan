import numpy as np

import logging
import urllib
import errno
import math
import sys
import os

from scipy.ndimage.measurements import label

DSPRITES_LINK = 'https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'


global dataset_zip
global imgs
global metadata
global image_size
global dataset_size

global latents_sizes
global latents_bases

# global reference to dataset
imgs = None
image_size = None
dataset_size = None

latents_sizes = None
latents_bases = None

logger = logging.getLogger(__name__)

def latent_to_index(latents):
    return np.dot(latents, latents_bases).astype(int)


def sample_latent(size=1):
    samples = np.zeros((size, latents_sizes.size))
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)

    return samples


def download_dsprites(dest):
    logger.debug('Downloading original dataset ...')

    urllib.request.urlretrieve(DSPRITES_LINK, dest)

    logger.debug('Done!')


def prepare_dsprites(type, repetitions=None):

    # check if original set exists
    dataset_file = os.environ['HOME'] + '/.forkan/datasets/dsprites.npz'
    if not os.path.isfile(dataset_file):
        parent_dir = os.environ['HOME'] + '/.forkan/datasets/'
        try:
            os.makedirs(parent_dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(parent_dir):
                pass
        download_dsprites(dataset_file)

    if type == 'original':
        dataset_file = os.environ['HOME'] + '/.forkan/datasets/dsprites.npz'
    elif type == 'translation':
        dataset_file = os.environ['HOME'] + '/.forkan/datasets/dsprites_translation.npz'
        if not os.path.isfile(dataset_file):
            generate_dsprites_translation()
    elif type == 'translation_scale':
        dataset_file = os.environ['HOME'] + '/.forkan/datasets/dsprites_translation_scale.npz'
        if not os.path.isfile(dataset_file):
            generate_dsprites_translation(with_scale=True)
    elif type == 'duo':
        dataset_file = os.environ['HOME'] + '/.forkan/datasets/dsprites_duo.npz'
        if not os.path.isfile(dataset_file):
            generate_dsprites_duo()
    else:
        logger.error('Unknown dataset {}. Exiting.'.format(type))
        sys.exit(1)

    # try to load dataset
    try:
        dataset_zip = np.load(dataset_file, encoding='latin1')
    except:
        logger.error('Could not load {}. Exiting.'.format(dataset_file))
        sys.exit(1)

    global imgs
    global image_size
    global dataset_size

    # get images, metadata and size
    imgs = dataset_zip['imgs']
    image_size = imgs.shape[1]
    dataset_size = imgs.shape[0]

    # load only for original dataset
    if type == 'original':
        metadata = dataset_zip['metadata'][()]

        global latents_sizes
        global latents_bases

        # Define number of values per latents and functions to convert to indices
        latents_sizes = metadata['latents_sizes']
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))

    # because its so small, translational set can be repeated
    if type == 'translation' and repetitions is not None:
        imgs = np.repeat(imgs, repetitions, axis=0)
        dataset_size *= repetitions


def load_dsprites(type='original', size=-1, validation_set=None,
                  data_format='channels_last', repetitions=None):

    prepare_dsprites(type=type, repetitions=repetitions)

    global imgs
    global image_size
    global dataset_size

    # simple sanity checks for size and percentage ranges
    assert dataset_size >= size

    if validation_set is not None:
        assert validation_set >= 0.0
        assert validation_set <= 1.0

    # reshape dataset
    if data_format == 'channels_last':
        imgs = np.reshape(imgs, (dataset_size, image_size, image_size, 1))
    else:
        imgs = np.reshape(imgs, (dataset_size, 1, image_size, image_size))

    # sample subset
    if size != -1:
        # Sample latents randomly
        latents_sampled = sample_latent(size=size)

        # Select images
        indices_sampled = latent_to_index(latents_sampled)
        imgs = imgs[indices_sampled]
        dataset_size = imgs.shape[0]

    if validation_set is None:
        return imgs, None
    else:
        cut = math.floor(dataset_size * validation_set)
        return imgs[cut:], imgs[:cut]


def load_dsprites_one_fixed(data_format='channels_last'):

    prepare_dsprites('original')

    # extract only translation latents
    latents = []
    for x in range(32):
                latents += [[0, 0, 0, 0, x, 0]]

    imgs_sampled = imgs[latent_to_index(latents)]

    # reshape and return
    if data_format == 'channels_last':
        return np.reshape(imgs_sampled, (-1, image_size, image_size, 1))
    else:
        return np.reshape(imgs_sampled, (-1, 1, image_size, image_size))


def generate_dsprites_duo():

    global imgs
    global loaded
    global image_size
    global dataset_size

    prepare_dsprites(type='original')

    dataset_dest = os.environ['HOME'] + '/.forkan/datasets/dsprites_duo.npz'

    # initialize output array
    duo = np.empty((0, image_size, image_size))

    logger.info('Generating duo dataset ...')

    # iterate over every second step of square and heart
    for bx in range(0, 32, 2):
        logger.debug('Step x {} with {} samples.'.format(bx//2, len(duo)))
        for by in range(0, 32, 2):
            logger.debug('Step y {} with {} samples.'.format(by// 2, len(duo)))

            # get base array
            bidx = latent_to_index([0, 0, 0, 0, bx, by])
            base = imgs[bidx]

            # create temporary array, otherwise concat would take hours
            tmp = np.empty((0, image_size, image_size))

            for ox in range(0, 32, 2):
                for oy in range(0, 32, 2):

                    # get array to overlay
                    oidx = latent_to_index([0, 2, 0, 0, ox, oy])
                    overlay = imgs[oidx]

                    # new array with 1 wherever one of the base arrays had one
                    merged = np.where(overlay != 0, overlay, base)

                    # find number of objects in new array
                    _, num_objects = label(merged)

                    if num_objects > 1:
                        tmp = np.concatenate((tmp, np.reshape(merged, (1, image_size, image_size))), axis=0)

            # add to main array
            duo = np.concatenate((duo, tmp), axis=0)

    logger.info('Successfully generated new dataset containing {} samples.\nSaving ...'.format(len(duo)))

    # saving dataset
    with open(dataset_dest, 'wb') as file:
        np.savez_compressed(file, imgs=duo)

    logger.info('Done.')


def generate_dsprites_translation(with_scale=False):

    global imgs
    global loaded
    global image_size
    global dataset_size

    prepare_dsprites(type='original')

    if not with_scale:
        dataset_dest = os.environ['HOME'] + '/.forkan/datasets/dsprites_translation.npz'
    else:
        dataset_dest = os.environ['HOME'] + '/.forkan/datasets/dsprites_translation_scale.npz'

    logger.info('Generating translation dataset ...')

    # extract only translational latents
    latents = []
    for shape in range(3):
        for scale in range(6):
            for x in range(32):
                for y in range(32):
                        latents += [[0, shape, scale, 0, x, y]]
            if not with_scale:
                break

    trans = imgs[latent_to_index(latents)]

    logger.info('Successfully generated new dataset containing {} samples.\nSaving ...'.format(len(trans)))

    # saving dataset
    with open(dataset_dest, 'wb') as file:
        np.savez_compressed(file, imgs=trans)

    logger.info('Done.')

if __name__ == '__main__':
    train, val = load_dsprites('translation')
    pass
