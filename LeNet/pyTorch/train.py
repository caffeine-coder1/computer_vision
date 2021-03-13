from model import LeNet
from data_pipe_line import get_data_loaders
from torch.optim import Adam, SGD
from general import set_logger, select_device, calculate_accuracy
from general import save_model, incremental_folder_name
from general import create_confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from torch.utils.tensorboard import SummaryWriter
import argparse
import time


def training(model, model_name, train, val, test, optimizer, criteria,
             tensorboard, epochs, device):
    """Training loop.

        Args:
            model: model to train (nn.Module object)
            model_name: model name for Tensor board entry.(str object)
            train: train set (DataLoader object)
            val: val set (DataLoader object)
            test: test set (DataLoader object)
            optimizer: torch.optim object.
            criteria: a loss function. should have backward() function.
            tensorboard: boolean value. True if you want to use tensorboard
            epochs: number of epochs to train.
            device: torch.device object
    """

    if tensorboard:
        summary_path = incremental_folder_name(
            base_dir='logs', folder=model_name)
        summary = SummaryWriter(summary_path)

    lowest_loss = float('inf')

    # ~~~~~~~~~~~~~~~~~~~~~ training loop ~~~~~~~~~~~~~~~~~~~~~ #

    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss, epoch_acc = 0, 0
        model.train()
        for x, y in train:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            # forward
            y_hat = model(x)
            loss = criteria(y_hat, y)
            loss.backward()

            # backward
            optimizer.step()

            with torch.no_grad():
                epoch_loss += loss
                acc = calculate_accuracy(y_hat, y)
                epoch_acc += acc

        epoch_loss = epoch_loss/len(train)
        epoch_acc = epoch_acc / len(train)

    # ~~~~~~~~~~~~~~~~~~~~~ eval loop ~~~~~~~~~~~~~~~~~~~~~ #

        val_loss, val_acc = 0, 0
        truth = []
        preds = []
        model.eval()
        with torch.no_grad():
            for x, y in val:
                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)

                y_prob = F.softmax(y_hat, dim=-1)
                pred = y_prob.argmax(1, keepdim=True)
                if device.type != 'cpu':
                    truth.append(y.cpu())
                    preds.append(pred.cpu())
                else:
                    truth.append(y)
                    preds.append(pred)

                loss = criteria(y_hat, y)
                val_loss += loss
                acc = calculate_accuracy(y_hat, y)
                val_acc += acc

            truth = torch.cat(truth, dim=0)
            preds = torch.cat(preds, dim=0)

            val_loss = val_loss/len(val)
            val_acc = val_acc / len(val)

    # ~~~~~~~~~~~~~~~~~~~~~ test loop ~~~~~~~~~~~~~~~~~~~~~ #

        if (epoch+1) % 5 == 0:
            test_loss, test_acc = 0, 0

            model.eval()
            with torch.no_grad():
                for x, y in test:
                    x = x.to(device)
                    y = y.to(device)

                    y_hat = model(x)
                    loss = criteria(y_hat, y)
                    test_loss += loss
                    acc = calculate_accuracy(y_hat, y)
                    test_acc += acc

                test_loss = test_loss/len(test)
                test_acc = test_acc / len(test)

    # ~~~~~~~~~~~~~~~~~~~~~ ploting and logging ~~~~~~~~~~~~~~~~~~~~~ #
        end_time = time.time()
        epoch_time = round(end_time - start_time, 2)
        logger.info(
            f'epoch:{epoch+1}|\t|train loss:{epoch_loss:.3f}| ' +
            f'|train acc:{epoch_acc:.3f}|\t|val loss:{val_loss:.3f} ' +
            f'|val acc:{val_acc:.3f}|\t|epoch time:{epoch_time} secs')

        if (epoch+1) % 5 == 0:
            logger.info(
                f'test:|\t\t\t|loss:{test_loss:.3f}|\t|acc:{test_acc:.3f}|')

        if tensorboard:
            summary.add_scalars(
                'loss', {'train': epoch_loss, 'val': val_loss},
                (epoch*len(train)))
            summary.add_scalars(
                'accuracy', {'train': epoch_acc, 'val': val_acc},
                (epoch*len(train)))
            confusion_matrix = create_confusion_matrix(truth, preds)
            summary.add_image(
                f'confusion_matrix/{model_name}', confusion_matrix,
                (epoch*len(train)))

    # ~~~~~~~~~~~~~~~~~~~~~ saving the weights ~~~~~~~~~~~~~~~~~~~~~ #

        if epoch == 0:
            weights_folder = incremental_folder_name(
                base_dir='weights', folder=model_name)

        lowest_loss = save_model(
            model, val_loss, lowest_loss, D=weights_folder)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='total batch size for all GPUs')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.09)
    parser.add_argument('--workers', type=int, default=2)

    opt = parser.parse_args()

    logger = set_logger('training')
    device = select_device({})

    # clearing the cache of cuda
    if device.type != 'cpu':
        # empty cuda cache
        logger.info('clearing torch cache...')
        torch.cuda.empty_cache()
        logger.info('torch cache cleared.')

    # clearing the python cache
    logger.info('clearing python cache...')
    gc.collect()
    logger.info('python cache cleared.')

    epochs = opt.epochs
    lr = opt.lr
    momentum = opt.momentum
    batch_size = opt.batch_size
    workers = opt.workers
    # get the dataset
    train, val, test = get_data_loaders(batch_size=batch_size, workers=workers)

    # list to store different models
    models = []
    # create model and move it to device
    lr_str = str(lr).replace('.', '')
    momentum_str = str(momentum).replace('.', '')

    model_name_adam = f'LeNet_adam_lr{lr_str}_bth_{batch_size}'
    model_adam = LeNet()
    model_adam = model_adam.to(device=device)

    # create optimizer
    optimizer_adm = Adam(model_adam.parameters(), lr=lr)

    # create loss
    loss_adm = nn.CrossEntropyLoss()
    loss_adm = loss_adm.to(device=device)

    models.append([model_name_adam, model_adam, optimizer_adm, loss_adm, True])
    # create model and move it to device
    model_name_sgd = f'LeNet_sgd_{lr_str}_bth_{batch_size}_m_{momentum_str}'
    model_sgd = LeNet()
    model_sgd = model_sgd.to(device=device)

    # create optimizer
    optimizer_sgd = SGD(model_sgd.parameters(), lr=lr, momentum=momentum)

    # create loss
    loss_sgd = nn.CrossEntropyLoss()
    loss_sgd = loss_sgd.to(device=device)

    models.append([model_name_sgd, model_sgd, optimizer_sgd, loss_sgd, True])

    for n, m, optim, loss, tb in models:

        logger.info(f'starting training for model: {n}')

        training(m, n, train, val, test,
                 optim, loss, tb, epochs, device)

        # clearing the cache of cuda
        if device.type != 'cpu':
            # empty cuda cache
            logger.info('clearing torch cache...')
            torch.cuda.empty_cache()
            logger.info('torch cache cleared.')

        # clearing the python cache
        logger.info('clearing python cache...')
        gc.collect()
        logger.info('python cache cleared.')
