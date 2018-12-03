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


def build_network(input_shape, num_actions, dueling=False, network_type='mini-mlp', scope='', reuse=None,
                  summaries=False):

    # use variable scope given and reuse is necessary
    with tf.variable_scope(scope, reuse=reuse):

        if network_type == 'mini-mlp':
            # add one dimension to shape
            input_shape = (None,) + input_shape

            # build standard MLP
            input_ = tf.placeholder(tf.float32, shape=input_shape, name='input')

            if not dueling:
                output = _mlp(input_, num_actions, [24, 24])
            else:
                # advantage value function
                A = _mlp(input_, num_actions, [24, 24])

                # state-value function
                V = _mlp(input_, num_actions, [24, 24])

                # plot mean(V(s, a))
                if summaries:
                    scalar_summary('state_value', tf.reduce_mean(V))

                # mean-center advantage values
                A_mean = tf.reduce_mean(A, axis=1)
                A_centered = A - tf.expand_dims(A_mean, axis=1)

                output = V + A_centered

        else:
            logger.critical('Network type {} unknown!'.format(network_type))
            sys.exit(0)

    return output, input_
