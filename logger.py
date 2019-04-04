import os
import logging


def setup_logger(name='', out_dir=''):
    # delete all root logger handles
    rootLogger = logging.getLogger()
    rootLogger.handlers = []

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # print(logging.Logger.manager.loggerDict)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(out_dir, name))
    fh.setLevel(logging.DEBUG)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # print(logger.handlers)
    return logger