#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D, Convolution2D
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

random.seed(42)
losses = {"d": [], "g": []}  # set up loss storage vector


class Generator(object):
    def __init__(self, nch, lr=0.0001):
        self.trainable = True
        self.nch = nch
        self.lr = lr
        g_in = Input(shape=[self.nch])
        h = Dense(nch * 14 * 14)(g_in)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Reshape([14, 14, nch])(h)
        h = UpSampling2D(size=(2, 2))(h)
        h = Conv2D(nch / 2, (3, 3), padding='same')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(nch / 4, (3, 3), padding='same')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(1, (1, 1), padding='same')(h)
        h = Activation('sigmoid')(h)
        h = Reshape([1, 28, 28])(h)
        self.model = Model(g_in, h)
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.lr))
        print("\nGenerator Summary:")
        self.model.summary()

    def change_lr(self, lr):
        self.lr = lr
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.lr))
        self.model.summary()

    def make_trainable(self, flag):
        self.trainable = flag
        for l in self.model.layers:
            l.trainable = self.trainable

    def plot(self, grid=(4, 4)):
        noise = np.random.uniform(0, 1, size=[grid[0] * grid[1], self.nch])
        gen_images = self.model.predict(noise)
        plt.figure(figsize=(np.sqrt(self.nch), np.sqrt(self.nch)))
        for i in range(gen_images.shape[0]):
            plt.subplot(grid[0], grid[1], i + 1)
            img = gen_images[i, 0, :, :]
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show()


class Discriminator(object):
    def __init__(self, s, dropout=0.2, lr=0.001):
        self.trainable = True
        self.dropout = dropout
        self.s = s
        self.lr = lr
        d_in = Input(shape=s)
        h = Conv2D(256, (5, 5), input_shape=s, strides=(2, 2), padding='same', activation='relu')(d_in)
        h = LeakyReLU(self.dropout)(h)  # use dropout for alpha
        h = Dropout(self.dropout)(h)
        h = Conv2D(512, (5, 5), strides=(2, 2), padding='same', activation='relu')(h)
        h = LeakyReLU(self.dropout)(h)
        h = Dropout(self.dropout)(h)
        h = Flatten()(h)
        h = Dense(256)(h)
        h = LeakyReLU(self.dropout)(h)
        h = Dropout(self.dropout)(h)
        h = Dense(2, activation='softmax')(h)
        self.model = Model(d_in, h)
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))
        print("\nDiscriminator Summary:")
        self.model.summary()

    def change_lr(self, lr):
        self.lr = lr
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))
        self.model.summary()

    def make_trainable(self, flag):
        self.trainable = flag
        for l in self.model.layers:
            l.trainable = self.trainable


class GAN(object):
    def __init__(self, nch, gen, dis, lr=0.0001):
        # Build stacked GAN model
        self.nch = nch
        self.lr = lr
        self.gan_in = Input(shape=[self.nch])
        h = gen.model(self.gan_in)
        gan_v = dis.model(h)
        self.model = Model(self.gan_in, gan_v)
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))
        print("\nGAN Summary:")
        self.model.summary()

    def change_lr(self, lr):
        self.lr = lr
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))
        self.model.summary()


def load_mnist_data(rows, cols):
    # load the data, shuffled and split between train and test sets
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    xtrain = xtrain.reshape(xtrain.shape[0], 1, rows, cols)
    xtest = xtest.reshape(xtest.shape[0], 1, rows, cols)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xtrain /= 255
    xtest /= 255
    return xtrain, ytrain, xtest, ytest


def plot_loss(l):
    plt.figure(figsize=(10, 8))
    plt.plot(l["d"], label='discriminitive loss')
    plt.plot(l["g"], label='generative loss')
    plt.legend()
    plt.show()


def plot_real(xtrain, n_ex=16, dim=(4, 4), figsize=(10, 10)):
    idx = np.random.randint(0, xtrain.shape[0], n_ex)
    gen_images = xtrain[idx, :, :, :]

    plt.figure(figsize=figsize)
    for i in range(gen_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        img = gen_images[i, 0, :, :]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Set up our main training loop
def train_for_n(gen, dis, gan, xtrain, nb_epoch=5000, plt_frq=25, batch_size=32, lr_g=0.0001, lr_d=0.001):
    for e in range(nb_epoch):
        # adapt learning rates
        gen.change_lr(lr_g)
        dis.change_lr(lr_d)

        # Make generative images
        image_batch = xtrain[np.random.randint(0, xtrain.shape[0], size=batch_size), :, :, :]
        noise_g = np.random.uniform(0, 1, size=[batch_size, 100])
        gen_images = gen.model.predict(noise_g)

        # Train discriminator on generated images
        x = np.concatenate((image_batch, gen_images))
        y = np.zeros([2 * batch_size, 2])
        y[0:batch_size, 1] = 1
        y[batch_size:, 0] = 1

        # make_trainable(discriminator,True)
        d_loss = dis.model.train_on_batch(x, y)
        losses["d"].append(d_loss)

        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0, 1, size=[batch_size, 100])
        y2 = np.zeros([batch_size, 2])
        y2[:, 1] = 1

        # make_trainable(discriminator,False)
        g_loss = gan.model.train_on_batch(noise_tr, y2)
        losses["g"].append(g_loss)

        # Updates plots
        if e % plt_frq == plt_frq - 1:
            plot_loss(losses)
            gen.plot()


def main():
    x_train, y_train, x_test, y_test = load_mnist_data(rows=28, cols=28)

    print(np.min(x_train), np.max(x_train))
    print('X_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # define models
    discriminator = Discriminator(x_train.shape[1:], dropout=0.2, lr=0.001)
    discriminator.make_trainable(False)
    generator = Generator(100)
    gan = GAN(100, generator, discriminator)

    ntrain = 10000

    trainidx = random.sample(range(0, x_train.shape[0]), ntrain)
    xt = x_train[trainidx, :, :, :]

    # Pre-train the discriminator network ...
    noise_gen = np.random.uniform(0, 1, size=[xt.shape[0], 100])
    generated_images = generator.model.predict(noise_gen)
    x = np.concatenate((xt, generated_images))
    n = xt.shape[0]
    y = np.zeros([2 * n, 2])
    y[:n, 1] = 1
    y[n:, 0] = 1

    discriminator.make_trainable(True)
    discriminator.model.fit(x, y, nb_epoch=1, batch_size=128)
    y_hat = discriminator.model.predict(x)

    # Measure accuracy of pre-trained discriminator network
    y_hat_idx = np.argmax(y_hat, axis=1)
    y_idx = np.argmax(y, axis=1)
    diff = y_idx - y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff == 0).sum()
    acc = n_rig * 100.0 / n_tot
    print("Accuracy: %0.02f pct (%d of %d) right" % (acc, n_rig, n_tot))

    # Train for 6000 epochs at original learning rates
    train_for_n(generator, discriminator, gan, x_train, nb_epoch=500, plt_frq=50, batch_size=32)

    # Train for 2000 epochs at reduced learning rates
    train_for_n(generator, discriminator, gan, x_train, nb_epoch=200, plt_frq=50, batch_size=32, lr_g=1e-5, lr_d=1e-4)

    # Train for 2000 epochs at reduced learning rates
    train_for_n(generator, discriminator, gan, x_train, nb_epoch=200, plt_frq=50, batch_size=32, lr_g=1e-6, lr_d=1e-5)

    # Plot the final loss curves
    plot_loss(losses)

    # Plot some generated images from our GAN
    generator.plot(grid=(5, 5))

    # Plot real MNIST images for comparison
    plot_real(x_train)


if __name__ == "__main__":
    main()
