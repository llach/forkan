import numpy as np
import matplotlib.pyplot as plt


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


def prune_dataset(set, batch_size):

    # keras training will die if there is a smaller batch at the end
    rest = set.shape[0] % batch_size
    if rest > 0:
        set = set[:-rest, ]

    return set


