import torch.nn.functional as F
import torch.nn as nn
from general import set_logger
from torchsummary import summary


class LeNet(nn.Module):
    """LeNet Model implementation in pyTorch. Total params 61K.

        Args:
            input size: (-1, 1, 32, 32) in (B, C, H, W) format.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.max_pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.max_pool(self.conv1(x))
        x = F.relu(x)
        x = self.max_pool(self.conv2(x))
        x = F.relu(x)
        x = x.view(-1, 16*5*5)
        x = self.fc3(self.fc2(self.fc1(x)))
        return x


if __name__ == '__main__':
    logger = set_logger(__name__)
    model = LeNet()
    summary(model, (1, 32, 32))
