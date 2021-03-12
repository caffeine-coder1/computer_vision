import torch
import numpy as np
import os
import random
import logging
from pathlib import Path

# ~~~~~~~~~~~~~~~~~~~~~ helper functions ~~~~~~~~~~~~~~~~~~~~~ #


def setSeed(seed):
    """Sets seed for random, numpy, and torch modules.

        Args:
            seed: a seed value. should be of type int.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def incrementalFolderName(base_dir='run', folder='train'):
    """Creates a Folder and returns its path in base_dir/folder{increment} format.

        Args:
            base_dir: base directory. this should be relative
                      to current python execution path.
            folder: the folder name that needs to be incremented.
                    like folder1, folder2, folder3.
    """
    i = 1
    while Path(f'{base_dir}/{folder}{i}').resolve().exists():
        i += 1
    p = Path(f'{base_dir}/{folder}{i}').resolve()
    p.mkdir()
    return str(p)


def incremental_filename(baseDir='run', name='predictions', ext='png'):
    """Returns a file path in base_dir/name{increment}.ext format.

        Args:
            base_dir: base directory. this should be relative
                      to current python execution path.
            name: the name of the file that needs to be incremented.
                  like file1, file2, file3.
    """

    i = 1
    while Path(f'{baseDir}/{name}{i}.{ext}').resolve().exists():
        i += 1
    fileName = Path(f'{baseDir}/{name}{i}.{ext}').resolve()
    return str(fileName)


def setConsoleLogger(loggerName):
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the  logger
    logging.getLogger(loggerName).addHandler(console)


def setFileLogger(loggerName):
    logFilePath = Path(f'logs/{loggerName}.log').resolve()
    if not logFilePath.exists():
        logFilePath.mkdir()
    # format for file logging
    formatter = logging.Formatter(
        fmt='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
    # file handler
    handler = logging.FileHandler(str(logFilePath), 'w')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger(loggerName).addHandler(handler)

# ~~~~~~~~~~~~~~~~~~~~~ logging functions ~~~~~~~~~~~~~~~~~~~~~ #


def set_logger(logger_name):
    """Creates a logger object with given name.

        Args:
                logger_name: str object, suggested usesage:
                             set_logger(__name__).
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    setConsoleLogger(logger_name)
    setFileLogger(logger_name)
    return logger


logger = set_logger(__name__)


# ~~~~~~~~~~~~~~~~~~~~~ network helper functions ~~~~~~~~~~~~~~~~~~~~~ #

def calculate_accuracy(y_hat, y):
    """Calculates the model accuracy.

        Args:
            y_hat: output of the Model. Argmax will be taken
                   in first dimension
            y: ground Truth. a 1D array of actual values.
    """
    top_pred = y_hat.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def save_model(model, current_loss, lowest_loss, D='weights/'):
    """Save the model to given directory in dir/run{increment} format

        Args:
            model: The model that needs to be saved.(nn.Module object)
            current_loss: loss at current state.(float object)
            lowest_loss: lowest loss seen till now. (float object)
            D: directory to save weights.
               default is '{current python execution}/weights/' (str object)
    """

    assert isinstance(D, str), ("expecting string object for path, " +
                                f'received object of {type(D)}')

    D = Path(D)
    if not D.exists():
        D.mkdir()
    last = (D/'last.pt').resolve()
    best = (D/'best.pt').resolve()

    torch.save(model.state_dict(), last)

    if current_loss < lowest_loss:
        torch.save(model.state_dict(), best)
        lowest_loss = current_loss

    return lowest_loss

# ~~~~~~~~~~~~~~~~~~~~~ system checks ~~~~~~~~~~~~~~~~~~~~~ #


def select_device(opt):
    """Returns a torch.device object based on the input.

        Args: 
             opt: a dictionary. should have a key called device.
                  for example: opt['device'] = 0 for 'cuda:0'
                  and 1 for 'cuda:1'.

                  incase of empty dictionary 'cuda:0' will be selected
                  if available.

                  in both the cases 'cpu' will be selected if cuda is
                  not available.
    """
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
