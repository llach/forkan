import numpy as np
import sys
import os

from scipy.ndimage.measurements import label


def generate_dsprites_duo(data_format='channels_last'):

    dataset_file = os.environ['HOME'] + '/.keras/datasets/dsprites.npz'
    dataset_dest = os.environ['HOME'] + '/.keras/datasets/dsprites_duo.npz'

    # try to load dataset
    try:
        print('Loading original dataset ...')
        dataset_zip = np.load(dataset_file, encoding='latin1')
    except:
        print('Could not find {}. Exiting.'.format(dataset_file))
        sys.exit(1)

    # get images, metadata and size
    imgs = dataset_zip['imgs']
    metadata = dataset_zip['metadata'][()]
    dataset_size = np.prod(metadata['latents_sizes'], axis=0)

    # image dimension
    idim = imgs.shape[1]

    # Define number of values per latents and functions to convert to indices
    latents_sizes = metadata['latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1, ])))

    # functions for random sampling copied from official repo
    def latent_to_index(latents):
        return np.dot(latents, latents_bases).astype(int)

    def sample_latent(size=1):
        samples = np.zeros((size, latents_sizes.size))
        for lat_i, lat_size in enumerate(latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)

    # initialize output array
    duo = np.empty((0, idim, idim))

    print('Generating new dataset ...')

    # iterate over every second step of square and heart
    for bx in range(0, 32, 2):
        print('Step x {} with {} samples.'.format(bx//2, len(duo)))
        for by in range(0, 32, 2):
            # print('Step y {} with {} samples.'.format(by// 2, len(duo)))

            # get base array
            bidx = latent_to_index([0, 0, 0, 0, bx, by])
            base = imgs[bidx]

            # create temporary array, otherwise concat would take hours
            tmp = np.empty((0, idim, idim))

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
                        tmp = np.concatenate((tmp, np.reshape(merged, (1, idim, idim))), axis=0)

            # add to main array
            duo = np.concatenate((duo, tmp), axis=0)

    print('Successfully generated new dataset containing {} samples.\nSaving ...'.format(len(duo)))

    # saving dataset
    with open(dataset_dest, 'wb') as file:
        np.savez_compressed(file, data=duo)

    print('Done.')


def generate_dsprites_translational(data_format='channels_last', with_scale=False):

    dataset_file = os.environ['HOME'] + '/.keras/datasets/dsprites.npz'

    if not with_scale:
        dataset_dest = os.environ['HOME'] + '/.keras/datasets/dsprites_translational.npz'
    else:
        dataset_dest = os.environ['HOME'] + '/.keras/datasets/dsprites_trans_scale.npz'

    # try to load dataset
    try:
        print('Loading original dataset ...')
        dataset_zip = np.load(dataset_file, encoding='latin1')
    except:
        print('Could not find {}. Exiting.'.format(dataset_file))
        sys.exit(1)

    # get images, metadata and size
    imgs = dataset_zip['imgs']
    metadata = dataset_zip['metadata'][()]
    dataset_size = np.prod(metadata['latents_sizes'], axis=0)

    # image dimension
    idim = imgs.shape[1]

    # Define number of values per latents and functions to convert to indices
    latents_sizes = metadata['latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1, ])))

    # functions for random sampling copied from official repo
    def latent_to_index(latents):
        return np.dot(latents, latents_bases).astype(int)

    print('Generating new dataset ...')

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

    print('Successfully generated new dataset containing {} samples.\nSaving ...'.format(len(trans)))

    # saving dataset
    with open(dataset_dest, 'wb') as file:
        np.savez_compressed(file, data=trans)

    print('Done.')

if __name__ == '__main__':
    generate_dsprites_translational(with_scale=True)
