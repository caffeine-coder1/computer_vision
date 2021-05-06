import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Linear
from torchsummary import summary


class yolo_model_traditional(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 64, 7, 2, 3)
        self.max_pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(64, 192, 3, 1, 1)
        self.conv3 = Conv2d(192, 128, 1, 1, 0)
        self.conv4 = Conv2d(128, 256, 3, 1, 1)
        self.conv5 = Conv2d(256, 256, 1, 1, 0)
        self.conv6 = Conv2d(256, 512, 3, 1, 1)

        self.conv7 = Conv2d(512, 256, 1, 1, 0)
        self.conv8 = Conv2d(256, 512, 3, 1, 1)

        self.conv9 = Conv2d(512, 512, 1, 1, 0)
        self.conv10 = Conv2d(512, 1024, 3, 1, 1)

        self.conv11 = Conv2d(1024, 512, 1, 1, 0)
        self.conv12 = Conv2d(1024, 1024, 3, 1, 1)
        self.conv13 = Conv2d(1024, 1024, 3, 2, 1)
        self.dense1 = Linear(7168, 4096)
        self.dense2 = Linear(4096, 1470)

    def forward(self, x):
        # block 1
        x = self.max_pool(self.conv1(x))

        # block 2
        x = self.max_pool(self.conv2(x))

        # block 3
        x = self.conv4(self.conv3(x))
        x = self.max_pool(self.conv6(self.conv5(x)))

        # block 4

        for _ in range(4):
            x = self.conv8(self.conv7(x))

        x = self.max_pool(self.conv10(self.conv9(x)))

        # block 5

        for _ in range(2):

            x = self.conv10(self.conv11(x))

        x = self.conv13(self.conv12(x))

        # block 6

        x = self.conv12(self.conv12(x))

        # dense layers

        x = x.view(-1, 7168)
        x = self.dense2(self.dense1(x))
        x = x.view(-1, 7, 7, 30)


if __name__ == "__main__":

    model = yolo_model_traditional()
    summary(model, (3, 448, 448))
