import os
import random

import numpy as np

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Dropout
from keras.optimizers import Adam, Adadelta
from keras import backend as K

from .base import BaseModel

from .utils import *
from .layers import *


class DiscriminatorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_fake)

        # num_to_switch = 15
        # y_pos = K.concatenate([y_pos[:-num_to_switch], y_neg[:num_to_switch]], axis=0)
        # y_neg = K.concatenate([y_neg[:-num_to_switch], y_pos[:num_to_switch]], axis=0)

        loss_real = keras.metrics.binary_crossentropy(y_pos, y_real)
        loss_fake = keras.metrics.binary_crossentropy(y_neg, y_fake)

        return K.mean(loss_real + loss_fake)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake = inputs[1]
        loss = self.lossfun(y_real, y_fake)
        self.add_loss(loss, inputs=inputs)

        return y_real


class GeneratorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_fake)

        loss_fake = keras.metrics.binary_crossentropy(y_pos, y_fake)
        loss_real = keras.metrics.binary_crossentropy(y_neg, y_real)

        return K.mean(loss_real + loss_fake)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake = inputs[1]
        loss = self.lossfun(y_real, y_fake)
        self.add_loss(loss, inputs=inputs)

        return y_real


def discriminator_accuracy(y_real, y_fake):
    y_pos = K.ones_like(y_real)
    y_neg = K.zeros_like(y_fake)
    acc_real = keras.metrics.binary_accuracy(y_pos, y_real)
    acc_fake = keras.metrics.binary_accuracy(y_neg, y_fake)
    return 0.5 * K.mean(acc_real + acc_fake)


def generator_accuracy(y_real, y_fake):
    y_pos = K.ones_like(y_fake)
    y_neg = K.zeros_like(y_real)
    acc_fake = keras.metrics.binary_accuracy(y_pos, y_fake)
    acc_real = keras.metrics.binary_accuracy(y_neg, y_real)
    return 0.5 * K.mean(acc_real + acc_fake)


class ALI(BaseModel):
    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='ali',
                 **kwargs
                 ):
        super(ALI, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_Gz = None
        self.f_Gx = None
        self.f_D = None

        self.gen_trainer = None
        self.dis_trainer = None

        self.build_model()

        self.last_d_loss = 10000000

        self.swap_prob = kwargs.get('swap_prob', 0.0)
        # save a checkpoint of the models each weights_checkpoint_depth iterations
        self.weights_checkpoint_depth = 20
        self.curr_checkpoint_iter = self.weights_checkpoint_depth
        self.swap_iter_length = 5
        self.gen_checkpoint_weights = None
        self.dis_checkpoint_weights = None
        self.gen_newest_weights = None
        self.dis_newest_weights = None

    def train_on_batch(self, x_real, compute_grad_norms=False):
        self.swap_weights()

        batchsize = len(x_real)
        y_pos, y_neg = ALI.get_labels(batchsize, self.label_smoothing)

        z_fake = np.random.normal(size=(batchsize, self.z_dims)).astype('float32')

        # indicates the upper for losses of the networks, i.e. a net will be retrained (although at most max_retrains
        # times) until the loss is lower than the bound
        max_loss = 5
        # also an upper bound but only for the generator network: the ratio of losses gen/dis (generator's loss can only
        # be max_g_2_d_loss_ratio times higher than discriminator's loss
        max_g_2_d_loss_ratio = 8.5
        retrained_times, max_retrains = 0, 20
        while True:
            g_loss, g_acc = self.gen_trainer.train_on_batch([x_real, z_fake], y_pos)

            if (g_loss < max_loss and g_loss < self.last_d_loss * max_g_2_d_loss_ratio) \
                    or retrained_times >= max_retrains:
                break
            retrained_times += 1
        if retrained_times > 0:
            print('Retrained Generator {} time(s)'.format(retrained_times))

        retrained_times = 0
        while True:
            d_loss, d_acc = self.dis_trainer.train_on_batch([x_real, z_fake], y_neg)

            if d_loss < max_loss or retrained_times >= max_retrains:
                break
            retrained_times += 1
        if retrained_times > 0:
            print('Retrained Discriminator {} time(s)'.format(retrained_times))
        self.last_d_loss = d_loss

        losses = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'g_acc': g_acc,
            'd_acc': d_acc
        }

        if compute_grad_norms:
            # https://stackoverflow.com/questions/45694344/calculating-gradient-norm-wrt-weights-with-keras
            grad_norm_func = get_gradient_norm_func(self.gen_trainer)
            gen_grad_norm = grad_norm_func([x_real, z_fake, np.ones(batchsize), y_pos.reshape((batchsize, 1)), 1])
            grad_norm_func = get_gradient_norm_func(self.dis_trainer)
            dis_grad_norm = grad_norm_func([x_real, z_fake, np.ones(batchsize), y_neg.reshape((batchsize, 1)), 1])
            losses['g_norm'] = gen_grad_norm[0]
            losses['d_norm'] = dis_grad_norm[0]

        return losses

    def predict(self, z_samples):
        return self.f_Gx.predict(z_samples)

    def build_model(self):
        self.f_Gz = self.build_Gz()
        self.f_Gx = self.build_Gx()
        self.f_D = self.build_D()

        self.f_Gz.summary()
        self.f_Gx.summary()
        self.f_D.summary()

        # Build discriminator
        set_trainable(self.f_Gz, False)
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)

        x_real = Input(shape=self.input_shape)
        z_fake = Input(shape=(self.z_dims,))

        x_fake = self.f_Gx(z_fake)
        z_avg, z_log_var = self.f_Gz(x_real)
        z_real = SampleNormal()([z_avg, z_log_var])

        y_real = self.f_D([x_real, z_real])
        y_fake = self.f_D([x_fake, z_fake])

        d_loss = DiscriminatorLossLayer()([y_real, y_fake])
        self.dis_trainer = Model([x_real, z_fake], d_loss)
        self.dis_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-5, beta_1=0.1),
                                 metrics=[discriminator_accuracy])
        self.dis_trainer.summary()

        # Build generators
        set_trainable(self.f_Gz, True)
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)

        g_loss = GeneratorLossLayer()([y_real, y_fake])
        self.gen_trainer = Model([x_real, z_fake], g_loss)
        self.gen_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-4, beta_1=0.5),
                                 metrics=[generator_accuracy])
        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')

    def build_Gz(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(inputs)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=512, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)

        x = Flatten()(x)
        z_avg = Dense(self.z_dims)(x)
        z_log_var = Dense(self.z_dims)(x)
        z_avg = Activation('linear')(z_avg)
        z_log_var = Activation('linear')(z_log_var)

        return Model(inputs, [z_avg, z_log_var])

    def build_Gx(self):
        inputs = Input(shape=(self.z_dims,))
        w = self.input_shape[0] // (2 ** 4)
        x = Dense(w * w * 512)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 512))(x)

        x = BasicDeconvLayer(filters=512, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicDeconvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicDeconvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicDeconvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, kernel_size=(3, 3), activation='tanh')(x)

        return Model(inputs, x)

    def build_D(self):
        x_inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x_inputs)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=512, kernel_size=(3, 3), bnorm=True)(x)
        x = Flatten()(x)

        z_inputs = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_inputs)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = Flatten()(z)

        xz = Concatenate(axis=-1)([x, z])
        xz = Dropout(0.2)(xz)
        xz = Dense(2048)(xz)
        xz = LeakyReLU(0.1)(xz)

        xz = Dropout(0.2)(xz)
        xz = Dense(1)(xz)
        xz = Activation('sigmoid')(xz)

        return Model([x_inputs, z_inputs], xz)

    def save_model(self, out_dir, epoch):
        self.trainers['f_D'] = self.f_D
        self.trainers['f_Gz'] = self.f_Gz
        super().save_model(out_dir, epoch)
        # remove f_dis from trainers to not load its weights when calling load_model()
        del self.trainers['f_D']
        del self.trainers['f_Gz']

    def swap_weights(self):
        if self.gen_newest_weights is not None or self.dis_newest_weights is not None:
            self.swap_iter_length -= 1
            if self.swap_iter_length == 0:
                if self.gen_newest_weights is not None:
                    print('Swapping generator weights back')
                    self.gen_trainer.set_weights(self.gen_newest_weights)
                    self.gen_newest_weights = None
                elif self.dis_newest_weights is not None:
                    print('Swapping discriminator weights back')
                    self.dis_trainer.set_weights(self.dis_newest_weights)
                    self.dis_newest_weights = None
                self.swap_iter_length = 5
        else:
            if self.curr_checkpoint_iter == self.weights_checkpoint_depth:
                # save checkpoints
                self.gen_checkpoint_weights = self.gen_trainer.get_weights()
                self.dis_checkpoint_weights = self.dis_trainer.get_weights()

                self.curr_checkpoint_iter = 0

            swap_gen = random.random() < self.swap_prob
            swap_dis = random.random() < self.swap_prob

            if swap_gen:
                print('Swapping generator weights')
                self.gen_newest_weights = self.gen_trainer.get_weights()
                self.gen_trainer.set_weights(self.gen_checkpoint_weights)
            elif swap_dis:
                print('Swapping discriminator weights')
                self.dis_newest_weights = self.dis_trainer.get_weights()
                self.dis_trainer.set_weights(self.dis_checkpoint_weights)

            self.curr_checkpoint_iter += 1
