import tensorflow as tf


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
