import os
import math
import errno
import numpy as np
import functools
import inspect
import logging
import matplotlib.pyplot as plt


from PIL import Image
from shutil import rmtree
from keras.utils import to_categorical

logger = logging.getLogger()


class textmod:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def textbf(text):
    return textmod.BOLD + text + textmod.END


def textul(text):
    return textmod.UNDERLINE + text + textmod.END


def textcolor(text, color='green'):

    if color == 'purple':
        return textmod.PURPLE + text + textmod.END
    elif color == 'cyan':
        return textmod.CYAN + text + textmod.END
    elif color == 'darkcyan':
        return textmod.DARKCYAN + text + textmod.END
    elif color == 'blue':
        return textmod.BLUE + text + textmod.END
    elif color == 'green':
        return textmod.GREEN + text + textmod.END
    elif color == 'yellow':
        return textmod.YELLOW + text + textmod.END
    elif color == 'red':
        return textmod.RED + text + textmod.END
    else:
        logger.warning('color {} no known. not modifying.'.format(color))
        return text


def print_dict(d, lo=None):
    lo = logger if lo is None else lo
    lo.info('{')
    for k, v in d.items():
        lo.info('     {}: {}'.format(k, v))
    lo.info('}')


def discount_with_dones(rewards, dones, gamma):
    """
    Calculates discounted rewards. This is still valid if the episode
    terminated within the sequence.

    From OpenAI basline's A2C.

    Parameters
    ----------
    rewards: list
        list of rewards with the last value being the state value

    dones: list
        list of dones as floats

    gamma: float
        discount factor

    Returns
    -------
        list of discounted rewards

    """
    discounted = []
    r = 0

    # discounted rewards are calculated on the reversed reward list.
    # that returns are calculated in descending order is easy to
    # overlook in the original pseudocode.
    # when writing down an example of the pseudocode, it is clear, that
    # r_t + gamma * V(s_tp1) is calculated for each list element and
    # this is also what is done here.
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def store_args(method):
    """Stores provided method args as instance attributes. From OpenAI baselines HER.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def create_dir(directory_path):
    if not os.path.isdir(directory_path):
        try:
            os.makedirs(directory_path)
            logger.info('Creating {}'.format(directory_path))
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory_path):
                pass


# get list of directories in path
def ls_dir(d):
    """ Returns list of subdirectories under d, excluding files. """
    return [d for d in [os.path.join(d, f) for f in os.listdir(d)] if os.path.isdir(d)]


def clean_dir(path):
    """ Deletes subdirs of path """

    logger.debug('Cleaning dir {}'.format(path))

    # sanity check
    if not os.path.isdir(path):
        return

    for di in ls_dir(path):
        rmtree(di)


def rename_latest_run(path):
    """ Renames 'run-latest' in path to run-ID """

    # sanity check
    if not os.path.isdir(path):
        return

    # for each found run-ID directory, increment idx
    idx = 1
    for di in ls_dir(path):
        if 'run-' in di:
            try:
                int(di.split('-')[-1])
                idx += 1
            except ValueError:
                continue

    # if a latest run exists, we rename it appropriately
    if os.path.isdir('{}/run-latest'.format(path)):
        os.rename('{}/run-latest'.format(path), '{}/run-{}'.format(path, idx))


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


def folder_to_unlabeled_npz(prefix, name, target_shape=None):
    logger.info('Converting {} to npz file.'.format(name))

    if target_shape is not None:
        logger.info('Dataset will have shape {}'.format(target_shape))

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
        if target_shape is None:
            imgs = np.array([np.array(Image.open(file_name), dtype=np.float32)
                             for file_name in file_paths], dtype=np.float32)
        else:
            imgs = np.array([np.array(Image.open(file_name).resize((target_shape[1], target_shape[0])), dtype=np.float32)
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
