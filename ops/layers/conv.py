import tensorflow as tf
from tensorflow.python.keras import layers as kl
from tensorflow.contrib import layers as tl
from ops.layers import activation


def conv1d(x, filters,
           kernel_size=3,
           stride=1,
           padding='same',
           activation_=None,
           is_training=True):
    """
    Args:
        x: input_tensor (N, L)
        filters: number of filters
        kernel_size: int
        stride: int
        padding: 'same' or 'valid'
        activation_: activation function
        is_training: True or False
    Returns:
    """
    with tf.variable_scope(None, conv1d.__name__):
        _x = tf.expand_dims(x, axis=2)
        _x = activation(kl.Conv2D(filters, (kernel_size, 1), (stride, 1), padding,
                                  activation=None, trainable=is_training)(_x),
                        activation_)
        _x = tf.squeeze(_x, axis=2)
    return _x


def conv1d_transpose(x, filters,
                     kernel_size=3,
                     stride=2,
                     padding='same',
                     activation_=None,
                     is_training=True):
    """
    Args:
        x: input_tensor (N, L)
        filters: number of filters
        kernel_size: int
        stride: int
        padding: 'same' or 'valid'
        activation_: activation function
        is_training: True or False
    Returns: tensor (N, L_)
    """
    with tf.variable_scope(None, conv1d_transpose.__name__):
        _x = tf.expand_dims(x, axis=2)
        _x = activation(kl.Conv2DTranspose(filters, (kernel_size, 1), (stride, 1), padding,
                                           activation=None, trainable=is_training)(_x),
                        activation_)
        _x = tf.squeeze(_x, axis=2)
    return _x


def conv2d(x, filters,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='same',
           activation_: str =None,
           kernel_initializer='glorot_uniform',
           bias_initializer='zeros',
           kernel_regularizer=None,
           bias_regularizer=None,
           is_training=True):
    return activation(kl.Conv2D(filters,
                                kernel_size,
                                strides,
                                padding,
                                activation=None,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                trainable=is_training)(x),
                      activation_)


def conv2d_transpose(x, filters,
                     kernel_size=(3, 3),
                     strides=(2, 2),
                     padding='same',
                     activation_=None,
                     kernel_initializer='glorot_uniform',
                     bias_initializer='zeros',
                     kernel_regularizer=None,
                     bias_regularizer=None,
                     is_training=True):
    return activation(kl.Conv2DTranspose(filters,
                                         kernel_size,
                                         strides,
                                         padding,
                                         activation=None,
                                         kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer,
                                         bias_regularizer=bias_regularizer,
                                         trainable=is_training)(x),
                      activation_)


def subpixel_conv2d(x, filters,
                    rate=2,
                    kernel_size=(3, 3),
                    activation_: str = None,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    is_training=True):
    with tf.variable_scope(None, subpixel_conv2d.__name__):
        _x = conv2d(x, filters*(rate**2),
                    kernel_size,
                    strides=(1, 1),
                    activation_=activation_,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    is_training=is_training)
        _x = pixel_shuffle(_x)
    return _x


def pixel_shuffle(x, r=2):
    with tf.name_scope(pixel_shuffle.__name__):
        return tf.depth_to_space(x, r)