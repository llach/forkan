import os
import tensorflow as tf

from shutil import rmtree


# get list of directories in path
def ls_dir(d):
    """ Returns list of subdirectories under d, excluding files. """
    return [d for d in [os.path.join(d, f) for f in os.listdir(d)] if os.path.isdir(d)]


def vector_summary(name, var, scope='vectors', with_hist=False):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    Copied from TensoFlow docs, but slightly modified.
    """
    with tf.name_scope('{}/{}'.format(scope, name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

        if with_hist:
            tf.summary.histogram('histogram', var)


def scalar_summary(name, var, scope='scalars'):
    """ Adds scalar Tensor to TensorBoard visualization under scope. """

    with tf.name_scope('{}/{}'.format(scope, name)):
        tf.summary.scalar(name, var)


def clean_dir(path):
    """ Deletes subdirs of path """

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
