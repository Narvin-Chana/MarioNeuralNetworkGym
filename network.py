import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, activations, regularizers, Model


def build_nn(data_size_in, n_classes):
    # This is a functional way of building the network which allows for more flexibility over a sequential one
    inputs = layers.Input(shape=data_size_in)

    # Different arguments can be passed to conv such as which activation, regularization, padding etc
    x_0 = layers.Conv2D(8, 2, strides=1, padding="same", activation="relu",
                        kernel_regularizer="l2")(inputs)

    x_1 = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu",
                        kernel_regularizer="l2")(x_0)

    x_2 = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu",
                        kernel_regularizer="l2")(x_1)

    x_3 = layers.Conv2D(16, 1, activation="relu")(x_2)
    x_4 = layers.Conv2D(2, 1, activation="relu")(x_3)
    x = layers.Flatten()(x_4)
    # Similar to conv. In this case I also passed an initializer, but conv can also take that argument
    dense = layers.Dense(n_classes, kernel_initializer="he_uniform", kernel_regularizer="l2")(x)
    # Normally speaking, leakyReLU is actually a better model to use which is why I show it here
    leak = layers.LeakyReLU()(dense)
    last = layers.Softmax()(leak)

    return Model(inputs=inputs, outputs=last)


def set_up_nn():
    n_actions = 12
    # Whole state for now, but 240 * 256 * 3 = 184.320,
    # so obviously want to find a more compact representation of environment
    data_size_in = (240, 256, 3)
    network = build_nn(data_size_in, n_actions)
    print(network.summary())
    return network
