from tensorflow.keras.datasets import mnist
import numpy as np

def load_mnist(flatten=True):
    # load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if flatten:
        # shape is (size, with, height)
        image_size = x_train.shape[1]

        # calculate
        original_dim = image_size * image_size

        # -1 inferrs shape from original one; reshapes without copying
        x_train = np.reshape(x_train, [-1, original_dim])
        x_test = np.reshape(x_test, [-1, original_dim])

        # normalize image values
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

    return (x_train, y_train), (x_test, y_test)

def prune_dataset(set, batch_size):

    # keras training will die if there is a smaller batch at the end
    rest = set.shape[0] % batch_size
    if rest > 0:
        set = set[:-rest, ]

    return set