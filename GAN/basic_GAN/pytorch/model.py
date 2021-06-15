import torch.nn as nn
from torchsummary import summary
import torch


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='r'):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        # ~~~~~~~~~~~~~~~~~~~ checks for activation ~~~~~~~~~~~~~~~~~~~#
        assert isinstance(activation, str), (
            'activation should be a str object' +
            f'not a type {type(activation)}')

        assert activation.lower() in ['r', 's', 't'], (
            'values accepted for activation are' +
            f' [r,s,t]. given {activation}')

        if activation.lower() == 'r':
            self.activation = nn.LeakyReLU()
        elif activation.lower() == 's':
            self.activation = nn.Sigmoid()

        else:
            self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.dense(x))


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        # input shape will be [N,1,32,32]
        self.net = nn.Sequential(
            nn.Flatten(), LinearBlock(img_size, 256),
            LinearBlock(256, 32), LinearBlock(32, 1, 's'))

    def forward(self, x):
        return self.net(x)


class Creator(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()

        # input shape will be [N,z_dim]
        self.net = nn.Sequential(
            LinearBlock(z_dim, 256), LinearBlock(256, img_size, 't'))

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':

    disc = Discriminator(1*28*28)
    gen = Creator(64, 1*28*28)

    # print(summary(disc, (1, 28, 28)))
    print(summary(gen, (64,)))
