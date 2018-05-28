from tensorflow.python.keras import Model
from ops.blocks import *
from ops.layers import conv2d


class Generator:
    def __init__(self, channel=3,
                 nb_growing=8,
                 upsampling_='upsampling'):
        self.name = 'models/generator'
        self.nb_growing = nb_growing
        self.filters = [512, 512, 512, 512, 256, 256, 128, 64, 32][:nb_growing]
        self.channel = channel
        self.upsampling = upsampling_

    def __call__(self, x,
                 reuse=False,
                 *args, **kwargs):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            o = None
            rgb_list = []
            for i, f in enumerate(self.filters):
                if i == 0:
                    g = Model(x, first_g_block(x, f), name='g_%d' % i)
                else:
                    g = Model(inputs=x,
                              outputs=g_block(o, f, self.upsampling),
                              name='g_%d' % i)
                o = g.output
                with tf.variable_scope(None, 'toRGB'):
                    rgb = conv2d(o, self.channel, activation_='tanh')
                rgb_list.append(rgb)
        return rgb_list

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]


class Discriminator:
    def __init__(self, channel=3,
                 nb_growing=8,
                 downsampling='average_pool'):
        self.name = 'models/discriminator'
        self.nb_growing = nb_growing
        self.filters = [512, 512, 512, 512, 256, 256, 128, 64, 32][:nb_growing][::-1]
        self.channel = channel
        self.downsampling = downsampling
        self.inputs = []
        self.inputs_ = []

    def __call__(self, rgb_list,
                 reuse=False,
                 *args, **kwargs):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            d_list = []
            o_list_ = []
            for i, f in enumerate(self.filters):
                if i == 0:
                    shape = (rgb_list[self.nb_growing-i-1].get_shape().as_list()[1],
                             rgb_list[self.nb_growing-i-1].get_shape().as_list()[2],
                             f)
                else:
                    shape = d_list[i-1].output_shape[1:]
                input_ = tf.keras.layers.Input(shape)
                if i == self.nb_growing-1:
                    d = Model(input_, last_d_block(input_, f), name='d_%d' % i)
                else:
                    d = Model(input_, d_block(input_, f), name='d_%d' % i)
                if not reuse:
                    self.inputs.append(input_)
                else:
                    self.inputs_.append(input_)
                d_list.append(d)
                o_list_.append(d.outputs[0])

            d_real = []
            for i in range(self.nb_growing):
                ds = d_list[i:]
                x = self.inputs[i]
                for d in ds:
                    x = d(x)
                d_real.append(x)

            d_fake = []
            for i in range(self.nb_growing):
                ds = d_list[i:]
                x = rgb_list[self.nb_growing-i-1]
                with tf.variable_scope(None, 'fromRGB'):
                    f = self.inputs[i].shape[-1]
                    x = conv2d(x, f, activation_='lrelu')
                for d in ds:
                    x = d(x)
                d_fake.append(x)
            return d_real, d_fake

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]


if __name__ == '__main__':
    _x = tf.keras.layers.Input((512, ), batch_size=2)
    _g = Generator(nb_growing=5, upsampling_='subpixel')
    _d = Discriminator(nb_growing=5)
    rgbs = _g(_x)
    _d_real, _d_fake = _d(rgbs)
    _d_real, _d_fake = _d(rgbs, reuse=True)
    sess = tf.keras.backend.get_session()
    tf.summary.FileWriter('./logs', graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    import numpy as np
    __x = np.random.uniform(size=(2, 4, 4, 512))
    print(sess.run(_d_real[-1],
                   feed_dict={_d.inputs[-1]: __x}))
