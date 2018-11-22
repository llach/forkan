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


def build_network(input_shape, num_actions, network_type='mlp', name=''):

    if network_type is 'mlp':
        with tf.name_scope(name):

            # add one dimension to shape
            input_shape = (None,) + input_shape

            # build standard MLP
            _input = tf.placeholder(tf.float32, shape=input_shape, name='{}/input'.format(name))
            dense1 = tf.layers.dense(_input, 64, name='{}/dense-1'.format(name),
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            dense2 = tf.layers.dense(dense1, 64, name='{}/dense-2'.format(name),
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            output = tf.layers.dense(dense2, num_actions, activation=None, name='{}/output'.format(name))

    else:
        logger.critical('Network type {} unknown!'.format(network_type))
        sys.exit(0)

    return output, _input
