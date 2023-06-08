import logging
from logging.handlers import RotatingFileHandler

MAX_BYTES = 10000000 # Maximum size for a log file

def initialize_logger(name: str, level: int):
    logger = logging.getLogger(f'{name}_console_log')
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)

    return logger

def initialize_file_logger(name: str, level: int):
    logger = logging.getLogger(f'{name}_file_log')
    logger.setLevel(level)

    file_handler = RotatingFileHandler('faster_whisper.log', maxBytes=MAX_BYTES)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":

    logger = initialize_logger(__name__, logging.DEBUG)
    file_logger = initialize_file_logger(__name__, logging.DEBUG)

    logger.info("test console logger only")
    file_logger.info("=======================================")
    file_logger.info("test file logger")
    file_logger.info("=======================================")