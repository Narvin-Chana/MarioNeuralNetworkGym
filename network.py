import tensorflow as tf
from tensorflow import keras
from keras import layers, models, activations, regularizers, Model


def build_nn(data_size_in, n_classes):
    """
    Build a small convolutional neural network
    :param data_size_in: shape of the incoming observation space
    :param n_classes: how many outputs the network produces
    :return: network
    """
    inputs = layers.Input(shape=data_size_in)

    x_0 = layers.Conv2D(8, 2, strides=1, activation="relu")(inputs)
    x_1 = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x_0)
    x_2 = layers.Conv2D(16, 3, strides=2, activation="relu")(x_1)

    x_3 = layers.Conv2D(16, 1, activation="relu")(x_2)
    x_4 = layers.Conv2D(2, 1, activation="relu")(x_3)
    x = layers.Flatten()(x_4)

    dense = layers.Dense(n_classes, kernel_initializer="he_uniform")(x)
    leak = layers.LeakyReLU()(dense)
    last = layers.Softmax()(leak)

    return Model(inputs=inputs, outputs=last)


def set_up_nn(n_actions):
    """
    Call to build a neural network
    :param n_actions: how many actions the network can output
    :return: constructed network
    """
    data_size_in = (15, 16, 4)
    network = build_nn(data_size_in, n_actions)
    print(network.summary())
    return network
