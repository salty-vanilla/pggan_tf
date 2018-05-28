import sys
import os
from tensorflow.python.keras.layers import Lambda
sys.path.append(os.getcwd())
from ops.layers import *


def first_g_block(x,
                  filters):
    with tf.variable_scope(None, first_g_block.__name__):
        x = reshape(x, (1, 1, x.get_shape().as_list()[-1]))
        x = upsampling2d(x, size=(4, 4))
        x = conv2d(x, filters,
                   activation_='lrelu',
                   kernel_initializer='normal')
        x = conv2d(x, filters,
                   activation_='lrelu',
                   kernel_initializer='normal')
        return x


def g_block(x,
            filters,
            upsampling_='upsampling'):
    with tf.variable_scope(None, g_block.__name__):
        if upsampling_ == 'subpixel':
            x = Lambda(lambda _x:
                       subpixel_conv2d(_x,
                                       filters,
                                       activation_='lrelu'))(x)
        elif upsampling_ == 'deconv':
            x = conv2d_transpose(x, filters, activation_='lrelu')
        elif upsampling_ == 'upsampling':
            x = upsampling2d(x)
        else:
            raise ValueError
        x = conv2d(x, filters,
                   activation_='lrelu',
                   kernel_initializer='normal')
        x = conv2d(x, filters,
                   activation_='lrelu',
                   kernel_initializer='normal')
        return x


def last_d_block(x,
                 filters):
    with tf.variable_scope(None, last_d_block.__name__):
        x = conv2d(x, filters,
                   activation_='lrelu',
                   kernel_initializer='normal')
        x = conv2d(x, filters, (4, 4),
                   activation_='lrelu',
                   padding='valid',
                   kernel_initializer='normal')
        return x


def d_block(x,
            filters,
            downsampling='average_pool'):
    with tf.variable_scope(None, d_block.__name__):
        x = conv2d(x, filters,
                   activation_='lrelu',
                   kernel_initializer='normal')
        if downsampling == 'average_pool':
            x = conv2d(x, filters,
                       activation_='lrelu',
                       kernel_initializer='normal')
            x = average_pool2d(x)
        elif downsampling == 'max_pool':
            x = conv2d(x, filters,
                       activation_='lrelu',
                       kernel_initializer='normal')
            x = max_pool2d(x)
        elif downsampling == 'stride':
            x = conv2d(x, filters,
                       activation_='lrelu',
                       strides=(2, 2),
                       kernel_initializer='normal')
        else:
            raise ValueError
        return x
