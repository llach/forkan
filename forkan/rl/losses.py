import tensorflow as tf


def huber_loss(a, delta=1.0):
    """
    Reference: https://en.wikipedia.org/wiki/Huber_loss
    Method from OpenAI baselines.
    """
    return tf.where(
        tf.abs(a) < delta,
        tf.square(a) * 0.5,
        delta * (tf.abs(a) - 0.5 * delta)
    )
