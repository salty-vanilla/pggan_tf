import tensorflow as tf
from tensorflow.python.keras import layers as kl
from tensorflow.contrib import layers as tl


def upsampling1d(x, size=2):
    """
    Args:
        x: input_tensor (N, L)
        size: int
    Returns: tensor (N, L*ks)
    """
    _x = tf.expand_dims(x, axis=2)
    _x = kl.UpSampling2D((size, 1))(_x)
    _x = tf.squeeze(_x, axis=2)
    return _x


def upsampling2d(x, size=(2, 2)):
    return kl.UpSampling2D(size)(x)