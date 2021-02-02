"""Loads a trained chemprop model checkpoint and makes predictions on a dataset."""

from chemprop.train import chemprop_predict
from loguru import logger

if __name__ == '__main__':
    logger.warning('Commencing prediction!')
    chemprop_predict()
