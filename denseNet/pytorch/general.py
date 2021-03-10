# ~~~~~~~~~~~~~~ helper functions for pyTorch ~~~~~~~~~~~~~~ #
import torch
import numpy as np
import random
from pathlib import Path
import logging
import os


def set_random_seed(SEED=1234):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def get_data_dir():
    return str(Path('./data/').resolve())


def setConsoleLogger(loggerName):
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the  logger
    logging.getLogger(loggerName).addHandler(console)


def setFileLogger(loggerName, write_mode):
    logFilePath = 'logs/%s.log' % (loggerName)
    if not os.path.exists('logs/'):
        os.makedirs('logs/')
    # format for file logging
    formatter = logging.Formatter(
        fmt='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
    # file handler
    handler = logging.FileHandler(logFilePath, write_mode)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger(loggerName).addHandler(handler)


def set_logger(loggerName, log_level=logging.DEBUG, write_mode='w'):
    logger = logging.getLogger(loggerName)
    logger.setLevel(log_level)
    setConsoleLogger(loggerName)
    setFileLogger(loggerName, write_mode)
    return logger


if __name__ == '__main__':
    pass
