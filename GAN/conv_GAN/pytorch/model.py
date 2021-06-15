import torch.nn as nn
from torchsummary import summary


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


class ConvTranspose2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU())

    def forward(self, x):
        return self.net(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_features, out_features, **kwargs),
            nn.BatchNorm2d(out_features), nn.LeakyReLU())

    def forward(self, x):
        return self.net(x)


class Conv1dBlock(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, out_features, **kwargs),
            nn.LeakyReLU())

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, in_features, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            Conv2DBlock(in_features, z_dim, kernel_size=3,
                        stride=2, padding=1, bias=False),
            Conv2DBlock(z_dim, z_dim, kernel_size=5, stride=2,
                        padding=0, bias=False),

            Conv2DBlock(z_dim, z_dim, kernel_size=5, stride=1,
                        padding=0, bias=False),
            nn.Flatten(), LinearBlock(1*1*z_dim, 1, 's'))

    def forward(self, x):
        return self.net(x)


class Faker(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(Conv2DBlock(
            z_dim, z_dim, kernel_size=3, stride=1, padding=1, bias=False),
            Conv2DBlock(
            z_dim, z_dim, kernel_size=3, stride=1, padding=1, bias=False),
            Conv2DBlock(
            z_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    disc = Discriminator(1, 32)
    faker = Faker(32)
    summary(disc, (1, 28, 28))
    summary(faker, (32, 28, 28))
