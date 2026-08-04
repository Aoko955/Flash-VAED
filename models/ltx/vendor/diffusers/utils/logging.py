import logging


def get_logger(name: str = "diffusers"):
    return logging.getLogger(name)
