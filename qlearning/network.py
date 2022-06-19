import tensorflow as tf
from tensorflow import keras
from keras import layers, models, activations, regularizers, Model


def build_nn(data_size_in, n_classes, is_target):
    """
    Build a small convolutional neural network
    :param data_size_in: shape of the incoming observation space
    :param n_classes: how many outputs the network produces
    :return: network
    """
    inputs = layers.Input(shape=data_size_in)

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(n_classes, activation="softmax")(layer5)

    if is_target:
        action = tf.stop_gradient(action)

    return keras.Model(inputs=inputs, outputs=action)


def set_up_nn(n_actions, is_target):
    """
    Call to build a neural network
    :param n_actions: how many actions the network can output
    :return: constructed network
    """
    data_size_in = (84, 84, 4)
    network = build_nn(data_size_in, n_actions, is_target)
    print(network.summary())
    return network
