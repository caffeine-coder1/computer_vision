import torch
from torch.nn import Conv2d, MaxPool2d, Linear, BatchNorm2d, Module
from torch.nn import LeakyReLU, BatchNorm1d, Sequential
from torchsummary import summary
import yaml


class BasicConvBlock(Module):

    def __init__(self, ic, oc, *kwargs):
        super().__init__()
        self.conv = Conv2d(ic, oc, bias=False, *kwargs)
        self.batch_norm = BatchNorm2d(oc)
        self.activation = LeakyReLU()

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv(x)))


class BasicLinearBlock(Module):

    def __init__(self, ic, oc, *kwargs):
        super().__init__()
        self.Linear = Linear(ic, oc, *kwargs)
        self.batch_norm = BatchNorm1d(oc)
        self.activation = LeakyReLU()

    def forward(self, x):
        return self.activation(self.batch_norm(self.Linear(x)))


class YOLOv1(Module):
    def __init__(self, channels, model_arch):
        super().__init__()
        self.channels = channels
        self.conv_layers = []
        self.dense_layers = []
        self.add_layers(model_arch)

    def add_layers(self, arch):
        in_channel = self.channels
        out_channel = 0
        for L in arch:
            if isinstance(L, tuple):
                if not isinstance(L[0], str):
                    out_channel = L[0]
                    self.conv_layers += [BasicConvBlock(in_channel,
                                                        out_channel,
                                                        L[1], L[2], L[3])]
                    in_channel = out_channel
                else:
                    if L[0].lower() == 'd':
                        self.dense_layers += [Linear(L[1], L[2])]
            if isinstance(L, str):
                if L.lower() == "m":
                    self.conv_layers += [MaxPool2d(2, 2)]

            if isinstance(L, list):
                for _ in range(L[-1]):
                    for r in range(len(L)-1):
                        out_channel = L[r][0]
                        self.conv_layers += [BasicConvBlock(in_channel,
                                                            out_channel,
                                                            L[r][1], L[r][2],
                                                            L[r][3])]
                        in_channel = out_channel

        self.conv_layers = Sequential(*self.conv_layers)
        self.dense_layers = Sequential(*self.dense_layers)

    def forward(self, x):

        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        print(x.shape)
        x = self.dense_layers(x)

        return x


if __name__ == "__main__":
    arch = []
    with open('model.yaml') as f:
        opt = yaml.full_load(f)
        arch = opt['architecture']
    model = YOLOv1(3, arch)
    summary(model, (3, 448, 448))
