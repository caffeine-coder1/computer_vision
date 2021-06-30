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
from general import print_gpu_details, print_memory_utilization


def training(opt):

    # ~~~~~~~~~~~~~~~~~~~ hyper parameters ~~~~~~~~~~~~~~~~~~~ #

    EPOCHS = opt.epochs
    CHANNELS = 1
    H, W = 64, 64
    work_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FEATURE_D = 128
    Z_DIM = 100
    BATCH_SIZE = opt.batch_size

    # ~~~~~~~~~~~~~~~~~~~ as per WGAN paper ~~~~~~~~~~~~~~~~~~~ #

    lr = opt.lr
    CRITIC_TRAIN_STEPS = 5
    WEIGHT_CLIP = 0.01

    # ~~~~~~~~~~~ creating directories for weights ~~~~~~~~~~~ #

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

    MNIST_data = MNIST(str(opt.data_dir), True, transform=trans, download=True)

    loader = DataLoader(MNIST_data, BATCH_SIZE, True,
                        num_workers=1, pin_memory=True,)

    # ~~~~~~~~~~~~~~~~~~~ creating tensorboard variables ~~~~~~~~~~~~~~~~~~~ #

    writer_fake = SummaryWriter(f"{str(log_dir)}/fake")
    writer_real = SummaryWriter(f"{str(log_dir)}/real")
    loss_writer = SummaryWriter(f"{str(log_dir)}/loss")

    # ~~~~~~~~~~~~~~~~~~~ loading the model ~~~~~~~~~~~~~~~~~~~ #

    critic = Discriminator(img_channels=CHANNELS,
                           feature_d=FEATURE_D).to(work_device)
    gen = Faker(Z_DIM, CHANNELS, FEATURE_D).to(work_device)

    if opt.resume:
        if Path(Weight_dir/'critic.pth').exists():

            critic.load_state_dict(torch.load(
                str(Weight_dir/'critic.pth'),
                map_location=work_device))

        if Path(Weight_dir/'generator.pth').exists():

            gen.load_state_dict(torch.load(
                str(Weight_dir/'generator.pth'),
                map_location=work_device))

    # ~~~~~~~~~~~~~~~~~~~ create optimizers ~~~~~~~~~~~~~~~~~~~ #

    critic_optim = optim.RMSprop(critic.parameters(), lr)
    gen_optim = optim.RMSprop(gen.parameters(), lr)

    # ~~~~~~~~~~~~~~~~~~~ training loop ~~~~~~~~~~~~~~~~~~~ #

    C_loss_prev = math.inf
    G_loss_prev = math.inf
    C_loss = 0
    G_loss = 0

    print_gpu_details()
    for epoch in range(EPOCHS):
        C_loss_avg = 0
        G_loss_avg = 0

        print_memory_utilization()

        for batch_idx, (real, _) in enumerate(tqdm(loader)):
            critic.train()
            gen.train()
            real = real.to(work_device)
            fixed_noise = torch.rand(
                real.shape[0], Z_DIM, 1, 1).to(work_device)

            # ~~~~~~~~~~~~~~~~~~~ discriminator loop ~~~~~~~~~~~~~~~~~~~ #

            for _ in range(CRITIC_TRAIN_STEPS):
                fake = gen(fixed_noise)  # dim of (N,1,W,H)

                # ~~~~~~~~~~~~~~~~~~~ forward ~~~~~~~~~~~~~~~~~~~ #

                # make it one dimensional array
                real_predict = critic(real).view(-1)
                # make it one dimensional array
                fake_predict = critic(fake).view(-1)

                # ~~~~~~~~~~~~~~~~~~~ loss ~~~~~~~~~~~~~~~~~~~ #

                C_loss = -(torch.mean(real_predict) - torch.mean(fake_predict))
                C_loss_avg += C_loss
                # ~~~~~~~~~~~~~~~~~~~ backward ~~~~~~~~~~~~~~~~~~~ #

                critic.zero_grad()
                C_loss.backward()
                critic_optim.step()

                # ~~~~~~~~~~~ weight cliping as per WGAN paper ~~~~~~~~~~ #

                for p in critic.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            # ~~~~~~~~~~~~~~~~~~~ generator loop ~~~~~~~~~~~~~~~~~~~ #

            fake = gen(fixed_noise)

            # ~~~~~~~~~~~~~~~~~~~ forward ~~~~~~~~~~~~~~~~~~~ #

            # make it one dimensional array
            fake_predict = critic(fake).view(-1)

            # ~~~~~~~~~~~~~~~~~~~ loss ~~~~~~~~~~~~~~~~~~~ #

            G_loss = -(torch.mean(fake_predict))
            G_loss_avg += G_loss
            # ~~~~~~~~~~~~~~~~~~~ backward ~~~~~~~~~~~~~~~~~~~ #

            gen.zero_grad()
            G_loss.backward()
            gen_optim.step()

            # ~~~~~~~~~~~~~~~~~~~ loading the tensorboard ~~~~~~~~~~~~~~~~~~~ #

            if batch_idx == 0 and epoch >= 1:
                C_loss_avg = C_loss_avg/(CRITIC_TRAIN_STEPS*BATCH_SIZE)
                G_loss_avg = G_loss_avg/(BATCH_SIZE)

                print(
                    f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(loader)}"
                    + f"Loss D: {C_loss_avg:.4f}, loss G: {G_loss_avg:.4f}"
                )

                with torch.no_grad():
                    critic.eval()
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
                        'Critic', C_loss, global_step=epoch)
                    loss_writer.add_scalar(
                        'generator', G_loss, global_step=epoch)

        # ~~~~~~~~~~~~~~~~~~~ saving the weights ~~~~~~~~~~~~~~~~~~~ #

        if opt.weights:
            if C_loss_prev > C_loss_avg:
                C_loss_prev = C_loss_avg
                weight_path = str(Weight_dir/'critic.pth')
                torch.save(critic.state_dict(), weight_path)

            if G_loss_prev > G_loss_avg:
                G_loss_prev = G_loss_avg
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
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate to use')

    parser.add_argument('--resume', type=bool, default=True,
                        help='should use the last saved weights')
    parser.add_argument('--data-dir', type=str, default='/datasets')
    opt = parser.parse_args()
    training(opt)
