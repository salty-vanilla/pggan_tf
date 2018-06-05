import tensorflow as tf
from tensorflow.python.keras.layers import Input
import numpy as np
import os
import csv
import time
from PIL import Image
from models import Generator, Discriminator
from ops.losses.gan import generator_loss, \
    discriminator_loss, \
    gradient_penalty, \
    discriminator_norm


class PGGAN:
    def __init__(self,
                 channel=3,
                 latent_dim=500,
                 nb_growing=8,
                 gp_lambda=10,
                 d_norm_eps=1e-3,
                 upsampling='subpixel',
                 downsampling='stride',
                 lr_d=1e-4,
                 lr_g=1e-4):
        self.channel = channel
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.nb_growing = nb_growing
        self.discriminator = Discriminator(nb_growing=nb_growing,
                                           downsampling=downsampling,
                                           channel=self.channel)
        self.generator = Generator(nb_growing=nb_growing,
                                   upsampling_=upsampling,
                                   channel=self.channel)
        self.latent_dim = latent_dim
        self.z = Input((self.latent_dim, ), name='z')
        self.bs = tf.placeholder(tf.int32, shape=[])

        self.gp_lambda = gp_lambda
        self.d_norm_eps = d_norm_eps

        self.sess = None
        self.saver = None
        self.fake = None

    def build_loss(self, inputs, growing_step):
        self.fake = self.generator(self.z,
                                   growing_step=growing_step)
        d_real = self.discriminator(inputs,
                                    growing_step=growing_step)
        d_fake = self.discriminator(self.fake,
                                    growing_step=growing_step,
                                    reuse=True)

        loss_g = generator_loss(d_fake, metrics='WD')
        loss_d = discriminator_loss(d_real, d_fake, metrics='WD')
        d_norm = discriminator_norm(d_real)

        # Gradient Penalty
        with tf.name_scope('GradientPenalty'):
            epsilon = tf.random_uniform(shape=[self.bs, 1, 1, 1],
                                        minval=0., maxval=1.)
            differences = self.fake - inputs
            interpolates = inputs + (epsilon * differences)
            gradients = tf.gradients(self.discriminator(interpolates,
                                                        growing_step=growing_step,
                                                        reuse=True), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gp = tf.reduce_mean(tf.square(slopes - 1.))

        return loss_g, loss_d, d_norm, gp

    def fit(self, image_sampler,
            noise_sampler,
            nb_epoch=1000,
            visualize_steps=1,
            save_steps=1,
            logdir='../logs'):

        model_path = None
        for growing_step in range(self.nb_growing):
            resolution = self.discriminator.resolutions[growing_step]
            image_sampler.target_size = resolution
            logdir_ = os.path.join(logdir, '{}_{}'.format(*resolution))
            os.makedirs(logdir_, exist_ok=True)
            inputs = tf.keras.Input((*resolution, self.channel))
            with tf.name_scope('Loss'):
                loss_g, loss_d, d_norm, gp = self.build_loss(inputs, growing_step)
                merged_loss_d = loss_d + self.gp_lambda*gp + self.d_norm_eps*d_norm

            with tf.variable_scope('Optimizer', reuse=tf.AUTO_REUSE):
                opt_d = tf.train.AdamOptimizer(learning_rate=self.lr_d, beta1=0.5, beta2=0.99) \
                    .minimize(merged_loss_d,
                              var_list=self.discriminator.vars)
                opt_g = tf.train.AdamOptimizer(learning_rate=self.lr_g, beta1=0.5, beta2=0.99) \
                    .minimize(loss_g,
                              var_list=self.generator.vars)

            with tf.variable_scope('Summary'):
                with tf.variable_scope('Generator'):
                    loss_g_summary = tf.summary.scalar('loss_g', loss_g)
                with tf.variable_scope('Discriminator'):
                    loss_d_summary = tf.summary.merge([tf.summary.scalar('loss_d', loss_d),
                                                       tf.summary.scalar('discriminator_norm', d_norm),
                                                       tf.summary.scalar('gradient_norm', gp)])
                image_summary = tf.summary.image('image', (self.fake+1)*0.5)
            if growing_step > 0:
                self.sess.run(tf.global_variables_initializer())
                self.restore(model_path)
            else:
                self.sess = tf.keras.backend.get_session()
                self.saver = tf.train.Saver(max_to_keep=None)
                self.sess.run(tf.global_variables_initializer())

            tb_writer = tf.summary.FileWriter(logdir_, graph=self.sess.graph)

            batch_size = image_sampler.batch_size
            nb_sample = image_sampler.nb_sample

            # calc steps_per_epoch
            steps_per_epoch = nb_sample // batch_size
            if nb_sample % batch_size != 0:
                steps_per_epoch += 1

            global_step = 0
            for epoch in range(1, nb_epoch + 1):
                print('\nepoch {} / {}'.format(epoch, nb_epoch))
                start = time.time()
                for iter_ in range(1, steps_per_epoch + 1):
                    image_batch = image_sampler()
                    noise_batch = noise_sampler(image_batch.shape[0], self.latent_dim)
                    if image_batch.shape[0] != batch_size:
                        continue
                    _, _loss_d, _gp, _d_norm, summary_d =\
                        self.sess.run([opt_d, loss_d, gp, d_norm, loss_d_summary],
                                      feed_dict={inputs: image_batch,
                                                 self.z: noise_batch,
                                                 self.bs: image_batch.shape[0]})
                    _, _loss_g, summary_g = self.sess.run([opt_g, loss_g, loss_g_summary],
                                                          feed_dict={self.z: noise_batch})

                    print('iter : {} / {}  {:.1f}[s]  loss_d : {:.4f}  loss_g : {:.4f}  \r'
                          .format(iter_, steps_per_epoch, time.time() - start,
                                  _loss_d, _loss_g), end='')
                    tb_writer.add_summary(summary_d, global_step)
                    tb_writer.add_summary(summary_g, global_step)
                    tb_writer.flush()
                    global_step += 1
                if epoch % save_steps == 0:
                    model_path = self.save(logdir_, epoch)
                if epoch % visualize_steps == 0:
                    noise_batch = noise_sampler(batch_size, self.latent_dim)
                    s = self.sess.run(image_summary,
                                      feed_dict={self.z: noise_batch})
                    tb_writer.add_summary(s, global_step)

                    # self.visualize()
        print('\nTraining is done ...\n')

    def restore(self, file_path):
        reader = tf.train.NewCheckpointReader(file_path)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0])
                            for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        var_dict = dict(zip(map(lambda x:
                                x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                current_var = var_dict[saved_var_name]
                var_shape = current_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(current_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(self.sess, file_path)

    def visualize(self, dst_dir, noise_batch, convert_function):
        generated_data = self.predict_on_batch(noise_batch)
        generated_images = convert_function(generated_data)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for i, image in enumerate(generated_images):
            if image.shape[2] == 1:
                image = image.reshape(image.shape[:2])
            dst_path = os.path.join(dst_dir, "{}.png".format(i))
            pil_image = Image.fromarray(np.uint8(image))
            pil_image.save(dst_path)

    def save(self, logdir, epoch):
        path = os.path.join(logdir, 'epoch_%d.ckpt' % epoch)
        self.saver.save(self.sess, save_path=path)
        return path

    def predict(self, x, batch_size=16):
        outputs = np.empty([0] + list(self.image_shape))
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            o = self.predict_on_batch(x_batch)
            outputs = np.append(outputs, o, axis=0)
        return outputs

    def predict_on_batch(self, x):
        return self.sess.run(self.fake,
                             feed_dict={self.z: x})
