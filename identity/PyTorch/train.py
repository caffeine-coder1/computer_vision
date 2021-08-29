from model import Identity
from data_pipe_line import TrafficDataSet, get_hash_matrix
from torch.optim import Adam
from general import set_logger, select_device, calculate_accuracy
from general import save_model, incremental_folder_name
from general import set_seed
import torch
from torch.utils.data import DataLoader
import gc
from torch.utils.tensorboard import SummaryWriter
import time
import yaml
from tqdm import tqdm
# from loss import TripletLoss
# ~~~~~~~~~~~~~~~~~~~~~ helper functions ~~~~~~~~~~~~~~~~~~~~~ #


def criteria(ai, pi, ni):
    global opt
    alpha = opt['alpha']
    positive_distance = torch.sum(torch.abs(ai-pi), dim=-1).squeeze()

    negative_distance = torch.sum(torch.abs(ai-ni), dim=-1).squeeze()

    loss = torch.sum((positive_distance - negative_distance) + alpha, dim=-1)
    return loss


def training(model, config, opt, train, optimizer, criteria, device):
    """Training loop.

        Args:
            model: model to train (nn.Module object)
            model_name: model name for Tensor board entry.(str object)
            train: train set (Dataset object)
            optimizer: torch.optim object.
            criteria: a loss function. should have backward() function.
            tensorboard: boolean value. True if you want to use tensorboard
            epochs: number of epochs to train.
            device: torch.device object
    """

    # ~~~~~~~~~~~~~~~~~ read and create parameters/objects ~~~~~~~~~~~~~~~~~ #

    model_name = config['model name']
    batch_size = config['batch size']
    classes = opt['classes']
    epochs = opt['epochs']
    tensorboard = opt['tensorboard']
    num_workers = opt['workers']
    hash_dir = opt['hash dir']
    alpha = opt['alpha']
    global lr
    model_name = f'{model_name}_bth_{batch_size}_epo_{epochs}_lr_{lr}'

    loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers)

    unknown_index = len(classes)-1

    if tensorboard:
        summary_path = incremental_folder_name(
            base_dir='logs', folder=model_name)
        summary = SummaryWriter(summary_path)

    lowest_acc = 0.0

    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss, epoch_acc = 0, 0

    # ~~~~~~~~~~~~~~~~~~~~~ training loop ~~~~~~~~~~~~~~~~~~~~~ #
        for ai, pi, ni, cl in tqdm(loader):
            model.train()
            ai = ai.to(device)
            pi = pi.to(device)
            ni = ni.to(device)
            optimizer.zero_grad()

            # forward
            ai_m = model(ai)
            pi_m = model(pi)
            ni_m = model(ni)
            loss = criteria(ai_m, pi_m, ni_m)

            # backward
            loss.backward()
            optimizer.step()

    # ~~~~~~~~~~~~~~~~~~~~~ predict loop ~~~~~~~~~~~~~~~~~~~~~ #

            with torch.no_grad():
                model.eval()
                x_hash = get_hash_matrix(model, hash_dir, device)
                # adding 1 in the first dimension for broadcasting.
                # now the dimension is [1,num_classes,1,128]
                x_hash = x_hash.unsqueeze(0)

                # adding 1 in the second dimension of ai  for broadcasting.
                ai_m = ai_m.unsqueeze(1)

                # calculate the norm matrix
                norm_matrix = torch.sum(
                    torch.abs(x_hash-ai_m), dim=-1).squeeze()

                # taking the min will give us the least L2 distance in the
                # last dimension
                L2_distance, index = torch.min(norm_matrix, -1)

                pred_class = torch.where(
                    L2_distance < alpha, index, unknown_index)

                # logger.info(f'{cl}')
                # logger.info(f'{pred_class}')

                epoch_loss += loss
                acc = calculate_accuracy(pred_class, cl)

                epoch_acc += acc

        epoch_loss = epoch_loss/len(train)
        epoch_acc = epoch_acc / len(train)

    # ~~~~~~~~~~~~~~~~~~~~~ update dataset ~~~~~~~~~~~~~~~~~~~~~ #

        if (epoch+1) % 5 == 0:
            train.update_dataset_random_choice()

    # ~~~~~~~~~~~~~~~~~~~~~ ploting and logging ~~~~~~~~~~~~~~~~~~~~~ #
        end_time = time.time()
        epoch_time = round(end_time - start_time, 2)
        logger.info(
            f'epoch:{epoch+1}|\t|train loss:{epoch_loss:.3f}| ' +
            f'|train acc:{epoch_acc:.3f}|' +
            f'\t|epoch time:{epoch_time} secs')

        if tensorboard:
            # if (epoch) % 5 == 0:
            #     confusion_matrix = create_confusion_matrix(
            #         class_list, pred_list, len(classes))
            #     summary.add_image(
            #         f'confusion_matrix/{model_name}', confusion_matrix,
            #         (epoch*len(train)))

            summary.add_scalars(
                'loss', {'train': epoch_loss},
                (epoch*len(train)))
            summary.add_scalars(
                'accuracy', {'val': epoch_acc},
                (epoch*len(train)))

    # ~~~~~~~~~~~~~~~~~~~~~ saving the weights ~~~~~~~~~~~~~~~~~~~~~ #

        if epoch == 0:
            weights_folder = incremental_folder_name(
                base_dir='identity/weights', folder=model_name)

        lowest_acc = save_model(
            model, epoch_acc, lowest_acc, D=weights_folder)

    # ~~~~~~~~~~~~~~~~~~~~~ clear cache ~~~~~~~~~~~~~~~~~~~~~ #

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


if __name__ == '__main__':

    opt = {}
    config = {}
    set_seed(12345)
    with open('identity/opt.yaml', 'r') as file:
        opt = yaml.full_load(file)

    with open('identity/config.yaml', 'r') as file:
        config = yaml.full_load(file)

    logger = set_logger('training')
    device = select_device(opt)

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

# ~~~~~~~~~~~~~~~~~~~~~ start training ~~~~~~~~~~~~~~~~~~~~~ #

    data_dir = opt['data dir']
    lr = config['lr']
    alpha = opt['alpha']
    model = Identity()
    data = TrafficDataSet(data_dir, model)
    optim = Adam(model.parameters(), lr)
    # loss = TripletLoss(alpha)

    training(model, config, opt, data, optim, criteria, device)

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
