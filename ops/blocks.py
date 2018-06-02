import sys
import os
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Lambda, Conv2D, UpSampling2D
sys.path.append(os.getcwd())
from ops.layers import *


class FirstGeneratorBlock(Model):
    def __init__(self, filters,
                 name=None):
        name = name if name is not None else self.__class__.__name__
        super().__init__(name=name)
        self.conv1 = Conv2D(filters, (3, 3),
                            padding='same',
                            kernel_initializer='he_normal')
        self.conv2 = Conv2D(filters, (3, 3),
                            padding='same',
                            kernel_initializer='he_normal')

    def call(self, inputs,
             training=None,
             mask=None):
        x = reshape(inputs, (1, 1, inputs.get_shape().as_list()[-1]))
        x = upsampling2d(x, (4, 4))
        x = self.conv1(x)
        x = activation(x, 'lrelu')
        x = pixel_norm(x)
        x = self.conv2(x)
        x = activation(x, 'lrelu')
        x = pixel_norm(x)
        return x


class GeneratorBlock(Model):
    def __init__(self, filters,
                 upsampling_='upsampling',
                 name=None):
        name = name if name is not None else self.__class__.__name__
        super().__init__(name=name)
        if upsampling_ == 'subpixel':
            self.up = Lambda(lambda _x:
                             subpixel_conv2d(_x,
                                             filters,
                                             activation_='lrelu'))
        elif upsampling_ == 'deconv':
            self.up = Lambda(lambda _x:
                             conv2d_transpose(_x,
                                              filters,
                                              activation_='lrelu'))
        elif upsampling_ == 'upsampling':
            self.up = UpSampling2D((2, 2))
        else:
            raise ValueError

        self.conv1 = Conv2D(filters, (3, 3),
                            padding='same',
                            kernel_initializer='he_normal')
        self.conv2 = Conv2D(filters, (3, 3),
                            padding='same',
                            kernel_initializer='he_normal')

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.up(inputs)
        x = self.conv1(x)
        x = activation(x, 'lrelu')
        x = pixel_norm(x)
        x = self.conv2(x)
        x = activation(x, 'lrelu')
        x = pixel_norm(x)
        return x


class LastDiscriminatorBlock(Model):
    def __init__(self, filters,
                 name=None):
        name = name if name is not None else self.__class__.__name__
        super().__init__(name=name)
        self.conv1 = Conv2D(filters, (3, 3),
                            padding='same',
                            kernel_initializer='he_normal')
        self.conv2 = Conv2D(filters, (4, 4),
                            padding='valid',
                            kernel_initializer='he_normal')

    def call(self, inputs,
             training=None,
             mask=None):
        x = minibatch_stddev(inputs, group_size=4)
        x = self.conv1(x)
        x = activation(x, 'lrelu')
        x = self.conv2(x)
        return x


class DiscriminatorBlock(Model):
    def __init__(self, filters_in,
                 filters_out,
                 downsampling='average_pool',
                 name=None):
        name = name if name is not None else self.__class__.__name__
        super().__init__(name=name)
        self.conv1 = Conv2D(filters_in, (3, 3),
                            padding='same',
                            kernel_initializer='he_normal')

        if downsampling == 'average_pool':
            self.down = Lambda(lambda _x: average_pool2d(conv2d(_x, filters_out,
                                                                activation_='lrelu',
                                                                kernel_initializer='he_normal')))
        elif downsampling == 'max_pool':
            self.down = Lambda(lambda _x: max_pool2d(conv2d(_x, filters_out,
                                                            activation_='lrelu',
                                                            kernel_initializer='he_normal')))
        elif downsampling == 'stride':
            self.down = Conv2D(filters_out, (3, 3),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer='he_normal')
        else:
            raise ValueError

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.conv1(inputs)
        x = activation(x, 'lrelu')
        x = self.down(x)
        return x
