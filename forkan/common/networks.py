import sys
import logging
import tensorflow as tf

from forkan.common.tf_utils import scalar_summary

"""
YETI - YEt To Implement

layer_norm: bool
        if true applies layer normalization for every layer
        as described in https://arxiv.org/abs/1607.06450
"""

logger = logging.getLogger(__name__)


def _mlp(input_, num_outputs, hiddens=[], output_activation=None):

    out = input_
    for h in hiddens:
        out = tf.contrib.layers.fully_connected(out, h, activation_fn=None)
        out = tf.nn.relu(out)
    return tf.contrib.layers.fully_connected(out, num_outputs, activation_fn=output_activation)


def _mlp_configured(input_, num_actions, hiddens, dueling=False, summaries=False):
    """ Constructs dueling architecture """

    # build standard MLP

    if not dueling:
        output = _mlp(input_, num_actions, hiddens)
    else:
        # advantage value function
        A = _mlp(input_, num_actions, hiddens)

        # state-value function
        V = _mlp(input_, num_actions, hiddens)

        # plot mean(V(s, a))
        if summaries:
            scalar_summary('state_value', tf.reduce_mean(V))

        # mean-center advantage values
        A_mean = tf.reduce_mean(A, axis=1)
        A_centered = A - tf.expand_dims(A_mean, axis=1)

        output = V + A_centered

    return output


def _nature_cnn(input_, num_actions, dueling=False, summaries=False):
    """ Returns CNN that Mnih et. al. described in the Nature paper """

    conv1 = tf.contrib.layers.conv2d(inputs=input_,
                                     num_outputs=32,
                                     kernel_size=(8, 8),
                                     stride=4,
                                     activation_fn=tf.nn.relu,)
    conv2 = tf.contrib.layers.conv2d(inputs=conv1,
                                     num_outputs=64,
                                     kernel_size=(4, 4),
                                     stride=1,
                                     activation_fn=tf.nn.relu,)
    flat = tf.layers.flatten(conv2)

    output = _mlp_configured(flat, num_actions, [512], dueling, summaries)

    return output


def build_network(input_shape, num_actions, dueling=False, network_type='mini-mlp', scope='', reuse=None,
                  summaries=False):

    # use variable scope given and reuse is necessary
    with tf.variable_scope(scope, reuse=reuse):

        # expand for batch dimension
        input_shape = (None,) + input_shape
        input_ = tf.placeholder(tf.float32, shape=input_shape, name='input')

        if network_type == 'mini-mlp':
            # squeeze away observation dimensions of 1
            if 1 in input_shape:
                squeeze_dims = []
                for i in range(len(input_shape)):
                    if input_shape[i] == 1:
                        squeeze_dims.append(i)
                input_ = tf.squeeze(input_, squeeze_dims)
            output = _mlp_configured(input_, num_actions, [24, 24], dueling, summaries)
        elif network_type == 'nature-cnn':
            output = _nature_cnn(input_, num_actions, dueling, summaries)
        else:
            logger.critical('Network type {} unknown!'.format(network_type))
            sys.exit(0)

    return input_, output
