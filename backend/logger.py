import logging

def initialize_logger(name: str, level: int):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)

    return logger