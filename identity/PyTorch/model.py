import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Identity(nn.Module):
    def __init__(self, n_chans1=64):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2a = nn.Conv2d(n_chans1, n_chans1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            n_chans1, 192, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=192)
        self.conv3a = nn.Conv2d(192, 192, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(384, 384, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6a = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    # concat code????
        self.fc1 = nn.Linear(7 * 7 * 256, 1 * 32 * 128)
        self.fc2 = nn.Linear(1 * 32 * 128, 1 * 32 * 128)
        self.fc7128 = nn.Linear(1 * 32 * 128, 1 * 1 * 128)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        # print(out.shape)# self method
        # out = self.resblocks(out)
        out = self.batch_norm1(out)
        # print(out.shape)
        out = F.torch.relu(self.conv2a(out))
        # print(out.shape)
        out = F.torch.relu(self.conv2(out))
        # print(out.shape)
        out = self.batch_norm2(out)
        # print(out.shape)
        out = F.max_pool2d(out, 2, padding=1)
        # print(out.shape)
        out = F.torch.relu(self.conv3a(out))
        # print(out.shape)
        out = F.torch.relu(self.conv3(out))
        # print(out.shape)
        out = F.max_pool2d(out, 2)
        # print(out.shape)
        out = F.torch.relu(self.conv4a(out))
        # print(out.shape)
        out = F.torch.relu(self.conv4(out))
        # print(out.shape)
        out = F.torch.relu(self.conv5a(out))
        # print(out.shape)
        out = F.torch.relu(self.conv5(out))
        # print(out.shape)
        out = F.torch.relu(self.conv6a(out))
        # print(out.shape)
        out = F.torch.relu(self.conv6(out))
        # print(out.shape)
        out = F.max_pool2d(out, 2)
        # print(out.shape)

        out = out.view(-1, 1, 7 * 7 * 256)
        # print(out.shape)
        out = torch.relu(self.fc1(out))
        # print(out.shape)
        out = torch.relu(self.fc2(out))
        # print(out.shape)
        out = torch.relu(self.fc7128(out))
        out = self.l2_norm(out)
        # print(out.shape)
        # out = torch.sqrt(torch.sum(out**2, dim=-1))
        # print(out.shape)
        return out


if __name__ == "__main__":
    model = Identity()
    summary(model, (3, 220, 220))
