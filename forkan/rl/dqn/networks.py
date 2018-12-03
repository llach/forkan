import sys
import logging
import tensorflow as tf

"""
YETI - YEt To Implement

layer_norm: bool
        if true applies layer normalization for every layer
        as described in https://arxiv.org/abs/1607.06450
"""

logger = logging.getLogger(__name__)


def build_network(input_shape, num_actions, network_type='mlp', scope='', reuse=False):

    # use variable scope given and reuse is necessary
    with tf.variable_scope(scope, reuse=reuse):

        if network_type is 'mlp':
            # add one dimension to shape
            input_shape = (None,) + input_shape

            # build standard MLP
            _input = tf.placeholder(tf.float32, shape=input_shape, name='input')
            dense1 = tf.contrib.layers.fully_connected(_input, 24)
            dense2 = tf.contrib.layers.fully_connected(dense1, 24)
            output = tf.contrib.layers.fully_connected(dense2, num_actions, activation_fn=None)

        else:
            logger.critical('Network type {} unknown!'.format(network_type))
            sys.exit(0)

    return output, _input
