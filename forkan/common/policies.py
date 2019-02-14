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

            with tf.variable_scope('logits', reuse=reuse):
                # one linear layer for action values, e.g. logits
                logits = tf.contrib.layers.fully_connected(mlp, num_actions, activation_fn=None)

            with tf.variable_scope('state_values', reuse=reuse):
                # linear activation with only one neuron representing the state value
                state_value = tf.contrib.layers.fully_connected(mlp, 1, activation_fn=None)

            with tf.variable_scope('action', reuse=reuse):
                # sample from categorical distribution
                u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
                action = tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)

            return input_, logits, state_value, action

        if policy_type == 'pi-and-value':

            # squeeze away observation dimensions of 1
            if 1 in input_shape:
                squeeze_dims = []
                for i in range(len(input_shape)):
                    if input_shape[i] == 1:
                        squeeze_dims.append(i)
                input_ = tf.squeeze(input_, squeeze_dims)

            # mlp base for policy
            with tf.variable_scope('policy'):
                pi_mlp = _fc(input_, [256, 256])
                logits = tf.contrib.layers.fully_connected(pi_mlp, num_actions, activation_fn=None)

                # sample from categorical distribution
                u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
                action = tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)

            with tf.variable_scope('value-function'):
                v_mlp = _fc(input_, [256, 256])
                state_value = tf.contrib.layers.fully_connected(v_mlp, 1, activation_fn=None)

            return input_, logits, state_value, action

        if policy_type == 'mnih-2013':

            conv1 = tf.contrib.layers.conv2d(input_, num_outputs=16, kernel_size=(8, 8), stride=4,
                                     activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=32, kernel_size=(4, 4), stride=2,
                                             activation_fn=tf.nn.relu)
            flat = tf.layers.flatten(conv2)
            mlp = _fc(flat, [256])

            # mlp base for policy
            with tf.variable_scope('policy'):

                logits = tf.contrib.layers.fully_connected(mlp, num_actions, activation_fn=None)

                # sample from categorical distribution
                u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
                action = tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)

            with tf.variable_scope('value-function'):
                state_value = tf.contrib.layers.fully_connected(mlp, 1, activation_fn=None)

            return input_, logits, state_value, action

        else:
            logger.critical('Policy type {} unknown!'.format(policy_type))
            sys.exit(0)

