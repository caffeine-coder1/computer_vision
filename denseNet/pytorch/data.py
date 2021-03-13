from general import get_data_dir
import torch
import torchvision.datasets as datasets
from torchvision import transforms


def get_MNIST_digit_data(dir):
    train_data = datasets.MNIST(root=dir,
                                train=True,
                                download=True, transform=transforms.ToTensor())
    val_data = datasets.MNIST(root=dir,
                              train=False,
                              download=True, transform=transforms.ToTensor())
    return train_data, val_data


if __name__ == '__main__':

    data_dir = get_data_dir()
    print(data_dir)
    train_data, val_data = get_MNIST_digit_data(data_dir)
