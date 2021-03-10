import torch
import numpy as np
import os
import random
import logging


# ~~~~~~~~~~~~~~~~~~~~~ helper functions ~~~~~~~~~~~~~~~~~~~~~ #


def setSeed():
    SEED = 98765
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def incrementalFolderName(baseDir='run', mode='train'):
    i = 1
    while os.path.exists(f'{baseDir}/{mode}{i}'):
        i += 1
    folderName = f'{baseDir}/{mode}{i}'
    os.mkdir(folderName)
    return folderName


def incremental_filename(baseDir='run', name='predictions', ext='png'):
    i = 1
    while os.path.exists(f'{baseDir}/{name}{i}.{ext}'):
        i += 1
    fileName = f'{baseDir}/{name}{i}.{ext}'
    return fileName


def setConsoleLogger(loggerName):
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the  logger
    logging.getLogger(loggerName).addHandler(console)


def setFileLogger(loggerName):
    logFilePath = 'logs/%s.log' % (loggerName)
    if not os.path.exists('logs/'):
        os.makedirs('logs/')
    # format for file logging
    formatter = logging.Formatter(
        fmt='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
    # file handler
    handler = logging.FileHandler(logFilePath, 'w')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger(loggerName).addHandler(handler)

# ~~~~~~~~~~~~~~~~~~~~~ logging functions ~~~~~~~~~~~~~~~~~~~~~ #


def set_logger(loggerName):
    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.DEBUG)
    setConsoleLogger(loggerName)
    setFileLogger(loggerName)
    return logger


logger = set_logger(__name__)

# ~~~~~~~~~~~~~~~~~~~~~ main functions call ~~~~~~~~~~~~~~~~~~~~~ #


def select_device(opt):
    # device = 'cpu' or '0' or '1'
    cpu = False
    s = 'cuda:0'
    if 'device' in opt.keys():  # device is in the opts
        device = opt['device']
        if device:  # device is not none
            cpu = device.lower() == 'cpu'  # device is cpu
        if device and not cpu:  # some device is requested and its not cpu
            cuda = torch.cuda.is_available()

            if cuda and int(device) in range(0, 5):
                # set environment variable
                os.environ['CUDA_VISIBLE_DEVICES'] = device
                s = f'cuda:{device}'
            else:

                logger.info(
                    f'CUDA unavailable or invalid device: {device} requested')
        elif cpu:
            # force torch.cuda.is_available() = False
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    cuda = torch.cuda.is_available() and not cpu
    if not cuda:
        logger.info('using cpu for computation.')
    else:
        logger.info(f'using {s} for computation.')
    return torch.device(s if cuda else 'cpu')
