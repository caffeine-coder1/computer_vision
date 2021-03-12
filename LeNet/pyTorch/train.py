from model import LeNet
from data_pipe_line import get_data_loaders
from torch.optim import Adam, SGD
from general import set_logger, select_device, calculate_accuracy
from general import save_model, incremental_folder_name
import torch
import torch.nn as nn
import gc
from torch.utils.tensorboard import SummaryWriter


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
    for epoch in range(epochs):
        epoch_loss, epoch_acc = 0, 0
        model.train()
        for x, y in train:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            y_hat = model(x)
            loss = criteria(y_hat, y)
            loss.backward()

            optimizer.step()
            with torch.no_grad():
                epoch_loss += loss
                acc = calculate_accuracy(y_hat, y)
                epoch_acc += acc

        epoch_loss = epoch_loss/len(train)
        epoch_acc = epoch_acc / len(train)
        logger.info(
            f'train|\t|epoch:{epoch}|\t|loss:{epoch_loss:.3f}|\t' +
            f'|acc:{epoch_acc:.3f}|')

        val_loss, val_acc = 0, 0
        model.eval()
        with torch.no_grad():
            for x, y in val:
                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                loss = criteria(y_hat, y)
                val_loss += loss
                acc = calculate_accuracy(y_hat, y)
                val_acc += acc

        val_loss = val_loss/len(val)
        val_acc = val_acc / len(val)

        logger.info(
            f'val|\t|epoch:{epoch}|\t|loss:{val_loss:.3f}|\t' +
            f'|acc:{val_acc:.3f}|')

        if tensorboard:
            summary.add_scalars(
                'loss', {'train': epoch_loss, 'val': val_loss},
                (epoch*len(train)))
            summary.add_scalars(
                'accuracy', {'train': epoch_acc, 'val': val_acc},
                (epoch*len(train)))

        if epoch == 0:
            weights_folder = incremental_folder_name(
                base_dir='weights', folder=model_name)

        lowest_loss = save_model(
            model, val_loss, lowest_loss, D=weights_folder)

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
            logger.info(
                f'test:|\t\t\t|loss:{test_loss:.3f}|\t|acc:{test_acc:.3f}|')


if __name__ == '__main__':

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

    epochs = 30
    lr = 0.001
    # get the dataset
    train, val, test = get_data_loaders(batch_size=1024)

    # create model and move it to device
    model_name_adam = 'LeNet_adam_lr0001'
    model_adam = LeNet()
    model_adam = model_adam.to(device=device)

    # create optimizer
    optimizer_adm = Adam(model_adam.parameters(), lr=lr)

    # create loss
    loss_adm = nn.CrossEntropyLoss()
    loss_adm = loss_adm.to(device=device)

    # create model and move it to device
    model_name_sgd = 'LeNet_sgd_lr0001_m09'
    model_sgd = LeNet()
    model_sgd = model_sgd.to(device=device)

    # create optimizer
    optimizer_sgd = SGD(model_sgd.parameters(), lr=lr, momentum=0.9)

    # create loss
    loss_sgd = nn.CrossEntropyLoss()
    loss_sgd = loss_sgd.to(device=device)

    training(model_adam, model_name_adam, train, val, test,
             optimizer_adm, loss_adm, True, epochs, device)

    training(model_sgd, model_name_sgd, train, val, test,
             optimizer_sgd, loss_sgd, True, epochs, device)
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
