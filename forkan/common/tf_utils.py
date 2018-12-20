import tensorflow as tf


def entropy_from_logits(logits):
    """
    Basically copied from OpenAIs PD classes, but with more comments in case
    anyone wants to understand whats going on.

    """

    # adding a constant in exp(x) to the x ( exp(x + a) ) is legit,
    # check https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
    # doing so will improve numerical stability (prevent infinities due to overflows)

    # some tensorflow gihtub issues also adress this issue: https://github.com/tensorflow/tensorflow/issues/2462
    # they also point to blog posts concerning this topic

    # this trick is further descirbed here: https://en.wikipedia.org/wiki/LogSumExp
    a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)

    # softmax on transformed logits
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0

    # entropy calculation with reversion of the max subtraction trick
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)


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
