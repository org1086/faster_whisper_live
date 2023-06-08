import logging
from logging.handlers import RotatingFileHandler

MAX_BYTES = 10000000 # Maximum size for a log file

def initialize_logger(name: str, level: int):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)

    return logger

def initialize_file_logger(name: str, level: int):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)

    file_log_format = logging.Formatter('%(message)s')

    file_handler = RotatingFileHandler('faster_whisper.log', maxBytes=MAX_BYTES)
    file_handler.setFormatter(file_log_format)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":

    file_logger = initialize_file_logger(__name__, logging.DEBUG)

    file_logger.info("=======================================")
    file_logger.info("test file logger")
    file_logger.info("=======================================")