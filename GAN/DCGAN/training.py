import argparse
import torch
import torchvision
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from model import Discriminator, Faker
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import math
from tqdm import tqdm
import shutil


def training(opt):

    # ~~~~~~~~~~~~~~~~~~~ hyper parameters ~~~~~~~~~~~~~~~~~~~ #
    EPOCHS = opt.epochs
    CHANNELS = 1
    H, W = 64, 64
    lr = opt.lr
    work_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FEATURE_D = 128
    Z_DIM = 100
    GEN_TRAIN_STEPS = 5
    BATCH_SIZE = opt.batch_size

    if opt.logs:
        log_dir = Path(f'{opt.logs}').resolve()
        if log_dir.exists():
            shutil.rmtree(str(log_dir))

    if opt.weights:
        Weight_dir = Path(f'{opt.weights}').resolve()
        if not Weight_dir.exists():
            Weight_dir.mkdir()
    # ~~~~~~~~~~~~~~~~~~~ loading the dataset ~~~~~~~~~~~~~~~~~~~ #

    trans = transforms.Compose(
        [transforms.Resize((H, W)),
         transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    MNIST_data = MNIST('./data', True, transform=trans, download=True)

    loader = DataLoader(MNIST_data, BATCH_SIZE, True, num_workers=1)

    # ~~~~~~~~~~~~~~~~~~~ creating tensorboard variables ~~~~~~~~~~~~~~~~~~~ #

    writer_fake = SummaryWriter(f"{str(log_dir)}/fake")
    writer_real = SummaryWriter(f"{str(log_dir)}/real")
    loss_writer = SummaryWriter(f"{str(log_dir)}/loss")

    # ~~~~~~~~~~~~~~~~~~~ loading the model ~~~~~~~~~~~~~~~~~~~ #

    disc = Discriminator(img_channels=CHANNELS,
                         feature_d=FEATURE_D).to(work_device)
    gen = Faker(Z_DIM, CHANNELS, FEATURE_D).to(work_device)

    if opt.resume:
        if Path(Weight_dir/'dirscriminator.pth').exists():

            disc.load_state_dict(torch.load(
                str(Weight_dir/'dirscriminator.pth'),
                map_location=work_device))

        if Path(Weight_dir/'generator.pth').exists():

            gen.load_state_dict(torch.load(
                str(Weight_dir/'generator.pth'),
                map_location=work_device))

    # ~~~~~~~~~~~~~~~~~~~ create optimizer and loss ~~~~~~~~~~~~~~~~~~~ #

    disc_optim = optim.Adam(disc.parameters(), lr, (0.5, 0.999))
    gen_optim = optim.Adam(gen.parameters(), lr, (0.5, 0.999))
    criterion = torch.nn.BCELoss()

    # ~~~~~~~~~~~~~~~~~~~ training loop ~~~~~~~~~~~~~~~~~~~ #
    D_loss_prev = math.inf
    G_loss_prev = math.inf
    D_loss = 0
    G_loss = 0

    for epoch in range(EPOCHS):

        for batch_idx, (real, _) in enumerate(tqdm(loader)):
            disc.train()
            gen.train()
            real = real.to(work_device)
            fixed_noise = torch.rand(
                real.shape[0], Z_DIM, 1, 1).to(work_device)
            # ~~~~~~~~~~~~~~~~~~~ discriminator loop ~~~~~~~~~~~~~~~~~~~ #

            fake = gen(fixed_noise)  # dim of (N,1,28,28)
            # ~~~~~~~~~~~~~~~~~~~ forward ~~~~~~~~~~~~~~~~~~~ #
            real_predict = disc(real).view(-1)  # make it one dimensional array
            fake_predict = disc(fake).view(-1)  # make it one dimensional array

            labels = torch.cat([torch.ones_like(real_predict),
                                torch.zeros_like(fake_predict)], dim=0)

            # ~~~~~~~~~~~~~~~~~~~ loss ~~~~~~~~~~~~~~~~~~~ #
            D_loss = criterion(
                torch.cat([real_predict, fake_predict], dim=0), labels)

            # ~~~~~~~~~~~~~~~~~~~ backward ~~~~~~~~~~~~~~~~~~~ #
            disc.zero_grad()
            D_loss.backward()
            disc_optim.step()

            # ~~~~~~~~~~~~~~~~~~~ generator loop ~~~~~~~~~~~~~~~~~~~ #
            for _ in range(GEN_TRAIN_STEPS):
                # ~~~~~~~~~~~~~~~~~~~ forward ~~~~~~~~~~~~~~~~~~~ #
                fake = gen(fixed_noise)
                # ~~~~~~~~~~~~~~~~~~~ forward ~~~~~~~~~~~~~~~~~~~ #
                # make it one dimensional array
                fake_predict = disc(fake).view(-1)
                # ~~~~~~~~~~~~~~~~~~~ loss ~~~~~~~~~~~~~~~~~~~ #

                G_loss = criterion(fake_predict, torch.ones_like(fake_predict))
                # ~~~~~~~~~~~~~~~~~~~ backward ~~~~~~~~~~~~~~~~~~~ #
                gen.zero_grad()
                G_loss.backward()
                gen_optim.step()

            # ~~~~~~~~~~~~~~~~~~~ loading the tensorboard ~~~~~~~~~~~~~~~~~~~ #

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(loader)} \
                                Loss D: {D_loss:.4f}, loss G: {G_loss:.4f}"
                )

                with torch.no_grad():
                    disc.eval()
                    gen.eval()
                    fake = gen(fixed_noise).reshape(-1, CHANNELS, H, W)
                    data = real.reshape(-1, CHANNELS, H, W)
                    if BATCH_SIZE > 32:
                        fake = fake[:32]
                        data = data[:32]
                    img_grid_fake = torchvision.utils.make_grid(
                        fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(
                        data, normalize=True)

                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=epoch
                    )
                    writer_real.add_image(
                        "Mnist Real Images", img_grid_real, global_step=epoch
                    )
                    loss_writer.add_scalar(
                        'discriminator', D_loss, global_step=epoch)
                    loss_writer.add_scalar(
                        'generator', G_loss, global_step=epoch)

        # ~~~~~~~~~~~~~~~~~~~ saving the weights ~~~~~~~~~~~~~~~~~~~ #
        if opt.weights:
            if D_loss_prev > D_loss:
                D_loss_prev = D_loss
                weight_path = str(Weight_dir/'dirscriminator.pth')
                torch.save(disc.state_dict(), weight_path)

            if G_loss_prev > G_loss:
                G_loss_prev = G_loss
                weight_path = str(Weight_dir/'generator.pth')
                torch.save(gen.state_dict(), weight_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # initial pre training weights
    parser.add_argument('--weights', type=str,
                        default='', help='save and load location of weights')
    parser.add_argument('--logs', type=str,
                        default='', help='save log files to')
    parser.add_argument("--epochs", type=int, default=20,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='total batch size for all GPUs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate to use')

    parser.add_argument('--resume', type=bool, default=True,
                        help='should use the last saved weights')
    opt = parser.parse_args()
    training(opt)
