from ops.blocks import *
from ops.layers import conv2d
from tensorflow.python.keras.layers import Dense


class Generator:
    def __init__(self, channel=3,
                 nb_growing=8,
                 upsampling_='upsampling'):
        self.name = 'models/generator'
        self.nb_growing = nb_growing
        self.filters = [512, 512, 512, 512, 256, 256, 128, 64, 32][:nb_growing]
        self.channel = channel
        self.upsampling = upsampling_

        self.blocks = []
        with tf.variable_scope(self.name):
            for i, f in enumerate(self.filters):
                if i == 0:
                    self.blocks.append(FirstGeneratorBlock(f, name='block_%d' % i))
                else:
                    self.blocks.append(GeneratorBlock(f, upsampling_, name='block_%d' % i))

    def __call__(self, x,
                 growing_step,
                 reuse=False,
                 *args, **kwargs):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            for block in self.blocks[:growing_step + 1]:
                x = block(inputs=x)
            with tf.variable_scope('toRGB_%d' % growing_step):
                x = conv2d(x, self.channel, activation_='tanh')
                return x

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]


class Discriminator:
    def __init__(self, channel=3,
                 nb_growing=8,
                 downsampling='average_pool'):
        self.name = 'models/discriminator'
        self.nb_growing = nb_growing
        self.filters = [512, 512, 512, 512, 256, 256, 128, 64, 32][:nb_growing]
        self.resolutions = [(2**(2+i), 2**(2+i)) for i in range(nb_growing)]
        self.channel = channel
        self.downsampling = downsampling
        self.blocks = []

        with tf.variable_scope(self.name):
            for i in range(nb_growing):
                if i == 0:
                    self.blocks.append(LastDiscriminatorBlock(self.filters[i], name='block_%d' % i))
                else:
                    self.blocks.append(DiscriminatorBlock(self.filters[i], self.filters[i-1],
                                                          downsampling, name='block_%d' % i))
            self.dense = Dense(1)

    def __call__(self, x,
                 growing_step,
                 reuse=False,
                 *args, **kwargs):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            with tf.variable_scope('fromRGB_%d' % growing_step):
                f = self.filters[growing_step]
                x = conv2d(x, f, activation_='tanh')
            for block in self.blocks[:growing_step + 1][::-1]:
                x = block(inputs=x)
            x = flatten(x)
            x = self.dense(x)
            return x

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]


if __name__ == '__main__':
    _x = tf.keras.layers.Input((512, ), batch_size=2)
    _g = Generator(nb_growing=5)
    rgbs = [_g(_x, growing_index=i) for i in range(5)]
    _d = Discriminator(nb_growing=5)
    _d_res = [_d(tf.keras.layers.Input((*_d.resolutions[i], _d.filters[i])), growing_index=i)
              for i in range(5)]
    sess = tf.keras.backend.get_session()
    tf.summary.FileWriter('./logs', graph=sess.graph)
    for v in tf.global_variables():
        print(v)
