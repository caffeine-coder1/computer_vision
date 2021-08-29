from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch
from model import Identity
from torchvision.utils import save_image
from general import incremental_filename, check_path
import gc
from general import set_logger
from tqdm import tqdm
import numpy as np
logger = set_logger(__name__, mode='a')


# ~~~~~~~~~~~~~~~~~~~~~ image tranforms ~~~~~~~~~~~~~~~~~~~~~ #

train_t = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((220, 220)),
    transforms.ToTensor(),
    transforms.Normalize([0.3417, 0.3126, 0.3216],
                         [0.168, 0.1678, 0.178])
])


trans = transforms.Compose([
    transforms.Resize((220, 220)),
    transforms.ToTensor(),
    transforms.Normalize([0.3417, 0.3126, 0.3216],
                         [0.168, 0.1678, 0.178])
])

# ~~~~~~~~~~~~~~~~~~~~~ helper functions ~~~~~~~~~~~~~~~~~~~~~ #


def create_one_hot(n, idx):
    one_hot = torch.zeros(idx.shape[0], n)
    one_hot.scatter_(1, idx.unsqueeze(1), 1)
    return one_hot


class TrafficDataSet(Dataset):

    """Returns triplet images (anchor, positive, and Negative) along with the class
        in [ai,pi,ni,cl] format.
        Args:
                dir:Dataset directory. this should be readable by
                torchvision.datasets.ImageFolder instance.
                model: torch.nn.Module instance.
    """

    def __init__(self, dir, model):
        super().__init__()
        self.trans = transforms.Compose([
            transforms.Resize((220, 220)),
            transforms.ToTensor(),
            transforms.Normalize([0.3417, 0.3126, 0.3216],
                                 [0.168, 0.1678, 0.178])
        ])
        self.train_t = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((220, 220)),
            transforms.ToTensor(),
            transforms.Normalize([0.3417, 0.3126, 0.3216],
                                 [0.168, 0.1678, 0.178])
        ])

        self.data = ImageFolder(dir, self.trans)
        self.class_len = len(self.data.classes)
        self.loader = DataLoader(self.data, 128, True)
        self.dataset = []
        self.update_dataset_random_choice()

    def update_dataset_using_norm(self, model):
        dataset = []
        model.eval()
        with torch.no_grad():
            logger.info('started updating dataset')
            for img, label in self.loader:
                train_norm = model(img)
                p_hot = create_one_hot(self.class_len, label)
                n_hot = torch.abs(p_hot - 1)

                for cl in range(len(label)):
                    pi_entries = (label == cl).type(torch.uint8)

                    if pi_entries.sum() > 1:
                        c_p_hot = p_hot[:, cl]
                        c_n_hot = n_hot[:, cl]
                        pi_entries = torch.nonzero(
                            pi_entries, as_tuple=True)

                        for ai in pi_entries[0]:
                            norm_matrix = torch.sqrt(
                                torch.sum(
                                    (train_norm[ai]-train_norm)**2, dim=-1)
                            ).squeeze()

                            pi_score, pi = torch.max(
                                norm_matrix*c_p_hot, dim=-1)
                            assert torch.numel(
                                pi) == 1, (f'pi size is {torch.numel(pi)}.' +
                                           ' it should be 1')

                            norm_matrix_n = torch.where(
                                norm_matrix > pi_score, norm_matrix, torch.tensor(0.0))

                            ni = torch.argmin(norm_matrix_n*c_n_hot, dim=-1)
                            assert torch.numel(
                                ni) == 1, (f'ni size is {torch.numel(ni)}.' +
                                           ' it should be 1')

                            dataset.append([img[ai], img[pi], img[ni], cl])
                            logger.info(
                                f'class:{cl} |anchor:{ai} |positive:{pi}' +
                                f'|negative: {ni}')
        logger.info(f"dataset size:{len(dataset)}")

        self.dataset = dataset
        del dataset

    def update_dataset_random_choice(self):
        dataset = []
        for img, label in tqdm(self.loader):
            for cl in range(len(label)):
                pi_entries = np.array((label == cl).nonzero(as_tuple=True)[0])
                ni_entries = np.array((label != cl).nonzero(as_tuple=True)[0])

                if pi_entries.sum() > 1:

                    for ai in pi_entries:
                        pi = np.random.choice(pi_entries, 1)[0]

                        assert pi.size == 1, (f'pi size is {torch.numel(pi)}.' +
                                              ' it should be 1')

                        ni = np.random.choice(ni_entries, 1)[0]

                        assert ni.size == 1, (f'ni size is {torch.numel(ni)}.' +
                                              ' it should be 1')

                        # logger.info(
                        #     f'class:{cl} |anchor:{ai} |positive:{pi}' +
                        #     f'|negative: {ni}')
                        dataset.append([img[ai], img[pi], img[ni], cl])
        logger.info(f"dataset size:{len(dataset)}")

        self.dataset = dataset
        del dataset

    def __getitem__(self, idx):

        ai = self.dataset[idx][0]
        pi = self.dataset[idx][1]
        ni = self.dataset[idx][2]
        cl = self.dataset[idx][3]
        # ai = self.ai_transforms(ai)
        # pi = self.pi_transforms(pi)
        # ni = self.ni_transforms(ni)
        return [ai, pi, ni, cl]

    def __len__(self):
        return len(self.dataset)


def get_hash_matrix(model, dir, device):

    device = device
    check_path(dir)

    data = ImageFolder(dir, transform=trans)
    loader = DataLoader(data, 128, False)

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            y_hat = model(x)

            return y_hat


if __name__ == "__main__":

    dir = '/home/user/datasets/GTSRB'
    model = Identity()
    data = TrafficDataSet(dir, model)
    logger.info(
        'enter an index for saving the data point\n enter q to quit...\n')

    key = input("index number: ")

    while key != 'q':

        assert key.isnumeric(), f'expecting an integer got {type(key)}'
        key = int(key)
        if key >= 0 and key < len(data):
            img = torch.stack(data[key][:3], dim=0)
            file_name = incremental_filename('identity/data_point', 'entry')
            save_image(img, file_name)
            logger.info(f'file saved as {file_name}\n')
            key = input('next index:')
        else:
            logger.info(f'entered value: {key} is out of range.' +
                        f'current range is 0 to {len(data)-1}')
            key = input('next index:')

    gc.collect()
