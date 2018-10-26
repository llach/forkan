import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

from tensorflow.keras.datasets import mnist
from transform_dsprites import generate_dsprites_duo, generate_dsprites_translational


def animate_greyscale_dataset(dataset):
    '''
    Animates dataset using matplotlib.
    Animations get slower over time. (Beware of big datasets!)

    :param dataset: dataset to animate
    '''

    # reshape if necessary
    if len(dataset.shape) and dataset.shape[-1] == 1:
        dataset = np.reshape(dataset, dataset.shape[:3])

    for i in range(dataset.shape[0]):
        plt.imshow(dataset[i], cmap='Greys_r')
        plt.pause(.005)

    plt.show()


def show_density(imgs):
  _, ax = plt.subplots()
  ax.imshow(imgs.mean(axis=0), interpolation='nearest', cmap='Greys_r')
  ax.grid('off')
  ax.set_xticks([])
  ax.set_yticks([])


# Helper function to show images
def show_images_grid(imgs_, num_images=25):
  ncols = int(np.ceil(num_images**0.5))
  nrows = int(np.ceil(num_images / ncols))
  _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
  axes = axes.flatten()

  for ax_i, ax in enumerate(axes):
    if ax_i < num_images:
      ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
      ax.set_xticks([])
      ax.set_yticks([])
    else:
      ax.axis('off')


def load_mnist(flatten=False):
    # load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if flatten:
        # shape is (size, with, height, channels?)
        image_size = x_train.shape[1]

        # calculate
        original_dim = image_size * image_size

        # -1 inferrs shape from original one; reshapes without copying
        x_train = np.reshape(x_train, [-1, original_dim])
        x_test = np.reshape(x_test, [-1, original_dim])

    else:
        # shape is (size, with, height, channels?)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # -1 inferrs shape from original one; reshapes without copying
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

    # normalize image values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return (x_train, y_train), (x_test, y_test)


def load_dsprites(size=-1, validation_set=None, format='channels_last'):
    dataset_file = os.environ['HOME'] + '/.keras/datasets/dsprites.npz'

    # try to load dataset
    try:
        dataset_zip = np.load(dataset_file, encoding='latin1')
    except:
        print('Could not find {}. Exiting.'.format(dataset_file))
        sys.exit(1)

    # get images, metadata and size
    imgs = dataset_zip['imgs']
    metadata = dataset_zip['metadata'][()]
    dataset_size = np.prod(metadata['latents_sizes'], axis=0)

    # image dimension
    isize = imgs.shape[1]

    # dataset size
    dsize = np.prod(metadata['latents_sizes'], axis=0)

    # reshape dataset
    if format == 'channels_last':
        imgs = np.reshape(imgs, (dsize, isize, isize, 1))
    else:
        imgs = np.reshape(imgs, (dsize, 1, isize, isize))

    # simple sanity checks for size and percentage ranges
    assert dataset_size >= size

    if validation_set is not None:
        assert validation_set >= 0.0
        assert validation_set <= 1.0

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

        return samples

    # sample subset
    if size != -1:
        # Sample latents randomly
        latents_sampled = sample_latent(size=size)

        # Select images
        indices_sampled = latent_to_index(latents_sampled)
        imgs = dataset_zip['imgs'][indices_sampled]
        dataset_size = imgs.shape[0]

    if validation_set == None:
        return (imgs, None)
    else:
        cut = math.floor(dataset_size * validation_set)
        return(imgs[cut:], imgs[:cut])


def load_dsprites_duo(format='channels_last', validation_set=None):
    dataset_file = os.environ['HOME'] + '/.keras/datasets/dsprites_duo.npz'

    # try to load dataset
    try:
        dataset_zip = np.load(dataset_file, encoding='latin1')
    except:
        print('Could not find {}. Trying to generate dataset ...'.format(dataset_file))
        generate_dsprites_duo()
        dataset_zip = np.load(dataset_file, encoding='latin1')

    # get images
    imgs = dataset_zip['data']

    # image dimension
    isize = imgs.shape[1]

    # dataset size
    dsize = imgs.shape[0]

    # reshape dataset
    if format == 'channels_last':
        imgs = np.reshape(imgs, (dsize, isize, isize, 1))
    else:
        imgs = np.reshape(imgs, (dsize, 1, isize, isize))

    if validation_set == None:
        return (imgs, None)
    else:
        cut = math.floor(dsize * validation_set)
        return(imgs[cut:], imgs[:cut])


def load_dsprites_translational(format='channels_last', validation_set=None, with_scale=False):

    if not with_scale:
        dataset_file = os.environ['HOME'] + '/.keras/datasets/dsprites_translational.npz'
    else:
        dataset_file = os.environ['HOME'] + '/.keras/datasets/dsprites_trans_scale.npz'

    # try to load dataset
    try:
        dataset_zip = np.load(dataset_file, encoding='latin1')
    except:
        print('Could not find {}. Trying to generate dataset ...'.format(dataset_file))
        generate_dsprites_translational(with_scale=with_scale)
        dataset_zip = np.load(dataset_file, encoding='latin1')

    # get images
    imgs = dataset_zip['data']

    # image dimension
    isize = imgs.shape[1]

    # dataset size
    dsize = imgs.shape[0]

    # reshape dataset
    if format == 'channels_last':
        imgs = np.reshape(imgs, (dsize, isize, isize, 1))
    else:
        imgs = np.reshape(imgs, (dsize, 1, isize, isize))

    if validation_set == None:
        return (imgs, None)
    else:
        cut = math.floor(dsize * validation_set)
        return(imgs[cut:], imgs[:cut])


def prune_dataset(set, batch_size):

    # keras training will die if there is a smaller batch at the end
    rest = set.shape[0] % batch_size
    if rest > 0:
        set = set[:-rest, ]

    return set


def plot_ae_mnist_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()