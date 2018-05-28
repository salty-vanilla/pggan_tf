import tensorflow as tf
from tensorflow.python.keras import layers as kl
from tensorflow.contrib import layers as tl


def layer_norm(x, is_training=True):
    return tl.layer_norm(x, trainable=is_training)


def batch_norm(x, is_training=True):
    return tl.batch_norm(x,
                         scale=True,
                         updates_collections=None,
                         is_training=is_training)
