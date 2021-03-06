import gzip
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import torch
from torchvision.transforms import transforms
from PIL import Image
import copy
from general import select_device, set_logger
import gc
import torchvision

logger = set_logger(__name__)

train_transforms = transforms.Compose([
    transforms.Resize(38),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.13], std=[0.3081])
])

test_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.13], std=[0.3081])
])


class MNISTDataset(Dataset):
    """This class provides the required data sets with transforms applied to it
        when its 'getitem'  method is called.

        Args:
                dir: directory in which the pickle file is located.
                test: boolean value. True for testset, false for Trainset
                transforms: a compose or single instance of
                            Torchvision.transforms
    """

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
            logger.info(f'loading test set with {x_test.shape[0]} examples')
            img_list = []
            for img in x_test:
                img_list.append(Image.fromarray(img))
            return img_list, y_test
        else:
            logger.info(f'loading train set with {x_train.shape[0]} examples')
            img_list = []
            for img in x_train:
                img_list.append(Image.fromarray(img))
            return img_list, y_train

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        x = self.dataset[0][idx]
        y = torch.tensor(self.dataset[1][idx], dtype=torch.long)
        if self.transforms:
            x = self.transforms(x)
        return x, y


def random_split_data_set(dataset, r=0.9):
    train_n = int(len(dataset)*r)
    val_n = int(len(dataset)) - train_n
    train_d, val_d = random_split(dataset, [train_n, val_n])
    return train_d, val_d


def get_data_loaders(dir='data/mnist.pkl.gz', batch_size=64, workers=1):
    """Returns 3 DataLoader instances namely train, val and test data loaders.

            Args:
                    dir: directory in which the pickle file is located.
                    batch_size: How many samples you want to load at once.
    """
    data = MNISTDataset(dir=dir, transforms=train_transforms)
    train_data, val_data = random_split_data_set(data, r=0.9)
    test_data = MNISTDataset(dir=dir, test=True, transforms=test_transforms)

    val_data = copy.deepcopy(val_data)
    val_data.transforms = test_transforms

    train_data_loader = DataLoader(
        train_data, batch_size, True, num_workers=workers)
    val_data_loader = DataLoader(
        val_data, batch_size, False, num_workers=workers)
    test_data_loader = DataLoader(
        test_data, batch_size, False, num_workers=workers)

    return train_data_loader, val_data_loader, test_data_loader


if __name__ == '__main__':
    device = select_device({})
    train_data, val_data, test_data = get_data_loaders(batch_size=4)

    for x, y in train_data:
        x = x.to(device)
        y = y.to(device)
        print(x.shape)
        print(y.shape)
        torchvision.utils.save_image(x, 'train.png')
        break
    for x, y in val_data:
        x = x.to(device)
        y = y.to(device)
        print(x.shape)
        print(y.shape)
        torchvision.utils.save_image(x, 'val.png')
        break
    for x, y in test_data:
        x = x.to(device)
        y = y.to(device)
        print(x.shape)
        print(y.shape)
        torchvision.utils.save_image(x, 'test.png')
        break

    gc.collect()
