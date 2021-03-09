import gzip
import pickle
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
from torchvision.transforms import transforms
from PIL import Image


class MNISTDataset(Dataset):
    def __init__(self, dir='data/mnist.pkl.gz', test=False, transforms=None):
        super().__init__()
        self.dir = str(Path(dir).resolve())
        self.test = test
        self.transforms = transforms
        assert Path(self.dir).exists(), "path to the MNIST data is wrong"
        self.dataset = self.read_data()

    def read_data(self):
        with gzip.open(self.dir, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            (x_train, y_train), (x_test, y_test) = data
        if self.test:
            print(f'loading test set with {x_test.shape[0]} examples')
            return x_test, y_test
        else:
            print(f'loading train set with {x_train.shape[0]} examples')
            return x_train, y_train

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        x = Image.fromarray(self.dataset[0][idx])
        y = torch.tensor(self.dataset[1][idx]).unsqueeze(0)
        if self.transforms:
            x = self.transforms(x)
        return x, y


train_transforms = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomCrop(28, padding=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.13], std=[0.3081])
])

test_transforms = transforms.Compose([
    transforms.RandomCrop(28, padding=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.13], std=[0.3081])
])


def get_data_loaders(dir='data/mnist.pkl.gz', batch_size=64):
    train_data = MNISTDataset(dir=dir, transforms=train_transforms)
    test_data = MNISTDataset(dir=dir, test=True, transforms=test_transforms)
    train_data_loader = DataLoader(train_data, batch_size, True)
    test_data_loader = DataLoader(test_data, batch_size, False)
    return train_data_loader, test_data_loader


if __name__ == '__main__':
    train_data, test_data = get_data_loaders()

    for x, y in train_data:
        print(x.shape)
        print(y.shape)
        break
    for x, y in test_data:
        print(x.shape)
        print(y.shape)
        break
