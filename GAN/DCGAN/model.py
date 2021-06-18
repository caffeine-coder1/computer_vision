import torch.nn as nn
from torchsummary import summary


class Discriminator(nn.Module):
    def __init__(self, img_channels, feature_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_d,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            *self.__block(feature_d, feature_d*2,
                          kernel_size=4, stride=2, padding=1),
            *self.__block(feature_d*2, feature_d*4,
                          kernel_size=4, stride=2, padding=1),
            *self.__block(feature_d*4, feature_d*8,
                          kernel_size=4, stride=2, padding=1),
            nn.Conv2d(feature_d*8, 1,
                      kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )
        self.initialize_weights()

    def __block(self, in_channels, out_channels, **kwargs):
        return [nn.Conv2d(in_channels, out_channels, **kwargs, bias=False),
                nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2)]

    def initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0, 0.02)

    def forward(self, x):
        return self.net(x)


class Faker(nn.Module):

    def __init__(self, z_dim, img_channels, feature_d):
        super().__init__()
        self.net = nn.Sequential(
            *self.__block(z_dim, feature_d*8, kernel_size=4,
                          stride=2, padding=0),
            *self.__block(feature_d*8, feature_d*4, kernel_size=4,
                          stride=2, padding=1),
            *self.__block(feature_d*4, feature_d*2, kernel_size=4,
                          stride=2, padding=1),
            *self.__block(feature_d*2, feature_d, kernel_size=4,
                          stride=2, padding=1),
            nn.ConvTranspose2d(feature_d, img_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.initialize_weights()

    def __block(self, in_channels, out_channels, **kwargs):

        return [nn.ConvTranspose2d(in_channels, out_channels, **kwargs, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()]

    def initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0, 0.02)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    disc = Discriminator(1, 128)
    gen = Faker(100, 1, 128)

    summary(disc, (1, 64, 64))

    summary(gen, (100, 1, 1))
