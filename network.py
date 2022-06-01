import numpy as np
import torch
import torch.nn as nn


class DQNSolver(nn.Module):
    """
    Convolutional Neural Net with 3 conv layers and two linear layers
    """

    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[-1], out_channels=8, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        print(self)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, n_actions),
            nn.LeakyReLU(),
            nn.Softmax(dim=0)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


def set_up_nn(n_actions):
    """
    Call to build a neural network
    :param n_actions: how many actions the network can output
    :return: constructed network
    """
    data_size_in = (15, 16, 4)
    network = DQNSolver(data_size_in, n_actions)
    print(network)
    return network
