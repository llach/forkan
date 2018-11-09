import os
import math
import errno
import numpy as np
import logging
import matplotlib.pyplot as plt

from PIL import Image
from keras.utils import to_categorical

logger = logging.getLogger()

def create_dir(directory_path):
    if not os.path.isdir(directory_path):
        try:
            os.makedirs(directory_path)
            logger.info('Creating {}'.format(directory_path))
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory_path):
                pass

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


def folder_to_npz(prefix, name, target_size, test_set):

    logger.info('Converting {} to npz file.'.format(name))

    # arrays for train & test images and labels
    x_train = np.empty([0] + target_size)
    y_train = np.empty([0])

    x_test = np.empty([0] + target_size)
    y_test = np.empty([0])

    # index to label mappings
    idx2label = {}
    label2idx = {}

    # build dataset dir
    dataset_dir = '{}/{}/'.format(prefix, name)
    dataset_save = '{}/{}.npz'.format(prefix, name)

    # iterate over class directories
    for directory, subdir_list, file_list in os.walk(dataset_dir):

        # skip if parent dir
        if directory == dataset_dir:
            continue

        class_name = directory.split('/')[-1]
        logger.info('Found class {} with {} samples.'.format(class_name, len(file_list)))

        # store idx -> label mapping
        idx = len(idx2label)
        idx2label[idx] = class_name
        label2idx[class_name] = idx

        # temp array for faster concat
        class_imgs = np.empty([0] + target_size)

        for file_name in file_list:
            # build file path
            file_path = '{}/{}'.format(directory, file_name)

            # load and resize image to desired input shape
            img = Image.open(file_path).resize([target_size[0], target_size[1]])

            # reshape for concatonation
            i = np.reshape(img, [1] + target_size).astype(np.float)

            # normalise image values
            i /= 255

            # append to temporary image array
            class_imgs = np.concatenate((class_imgs, i))

        # split class into train & test images
        nb_test = math.floor(class_imgs.shape[0]*test_set)
        nb_train = class_imgs.shape[0] - nb_test

        logger.info('Splitting into {} train and {} test samples.'.format(nb_train, nb_test))

        # randomly shuffle dataset before splitting into train & test
        np.random.shuffle(class_imgs)

        # do the split
        train, test = class_imgs[:nb_train], class_imgs[nb_train:]

        # append to final image array
        x_train = np.concatenate((x_train, train))
        x_test = np.concatenate((x_test, test))
        
        # add labels
        y_train = np.concatenate((y_train, [idx] * nb_train))
        y_test = np.concatenate((y_test, [idx] * nb_test))

    # convert label vector to binary class matrix
    nb_classes = len(idx2label.keys())
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    logger.info('Dataset has {} train and {} test samples.'.format(x_train.shape[0], x_test.shape[0]))
    logger.info('Saving dataset ...')

    # save dataset
    with open(dataset_save, 'wb') as file:
        np.savez_compressed(file, x_train=x_train, y_train=y_train,
                            x_test=x_test, y_test=y_test,
                            idx2label=idx2label, label2idx=label2idx)

    logger.info('Done!')


def folder_to_unlabeled_npz(prefix, name):
    logger.info('Converting {} to npz file.'.format(name))

    # build dataset dir
    dataset_dir = '{}/{}/'.format(prefix, name)
    dataset_save = '{}/{}.npz'.format(prefix, name)

    for _, _, file_list in os.walk(dataset_dir):
        for file in file_list:
            im_path = os.path.join(dataset_dir, file)
            image_shape = list(np.array(Image.open(im_path)).shape)
            break
        break

    # iterate over class directories
    for directory, subdir_list, file_list in os.walk(dataset_dir):

        # build file path
        file_paths = ['{}/{}'.format(directory, file_name) for file_name in file_list]

        # load and resize image to desired input shape
        imgs = np.array([np.reshape(Image.open(file_name), image_shape)
                         for file_name in file_paths], dtype=np.float32)

        # normalise image values
        imgs /= 255

        # randomly shuffle dataset
        np.random.shuffle(imgs)

        break

    logger.info('Dataset has {} samples.'.format(imgs.shape[0]))
    logger.info('Saving dataset ...')

    # save dataset
    with open(dataset_save, 'wb') as file:
        np.savez_compressed(file, imgs=imgs)

    logger.info('Done!')
