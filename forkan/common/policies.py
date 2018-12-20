import sys
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


def _fc(input_, hiddens=[]):

    out = input_
    for h in hiddens:
        out = tf.contrib.layers.fully_connected(out, h, activation_fn=None)
        out = tf.nn.relu(out)
    return out


def build_policy(input_shape, num_actions, policy_type='mini-mlp', scope='', reuse=None):

    # use variable scope given and reuse is necessary
    with tf.variable_scope(scope, reuse=reuse):
        # expand for batch dimension
        input_shape = (None,) + input_shape
        input_ = tf.placeholder(tf.float32, shape=input_shape, name='input')

        if policy_type == 'mini-mlp':

            # squeeze away observation dimensions of 1
            if 1 in input_shape:
                squeeze_dims = []
                for i in range(len(input_shape)):
                    if input_shape[i] == 1:
                        squeeze_dims.append(i)
                input_ = tf.squeeze(input_, squeeze_dims)

            # small network with no output
            mlp = _fc(input_, [256, 256])

            with tf.variable_scope('action_values', reuse=reuse):
                # one linear layer for action values, e.g. logits
                action_values = tf.contrib.layers.fully_connected(mlp, num_actions, activation_fn=None)

            with tf.variable_scope('PI', reuse=reuse):
                # softmax of logits calculates PI(a)
                pi = tf.nn.softmax(action_values, axis=1)

            with tf.variable_scope('state_values', reuse=reuse):
                # linear activation with only one neuron representing the state value
                state_value = tf.contrib.layers.fully_connected(mlp, 1, activation_fn=None)

        else:
            logger.critical('Policy type {} unknown!'.format(policy_type))
            sys.exit(0)

    return input_, action_values, pi, state_value
