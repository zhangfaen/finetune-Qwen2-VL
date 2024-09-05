import logging
import datetime
import os

# Create a logger object
_logger = None

def init_logger(log_dir="./"):
    os.makedirs(log_dir, exist_ok=True)
    global _logger
    _logger = logging.getLogger('MyLogger')
    _logger.setLevel(logging.INFO)  # Set the default logging level to INFO

    # Create a formatter with detailed format including filename and line number
    _formatter = logging.Formatter('%(asctime)s-%(filename)s:%(lineno)d-%(levelname)s >> %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create a file handler and set the level to INFO
    _file_handler = logging.FileHandler(f'{log_dir}output.{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log.txt', mode='w')
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(_formatter)

    # Create a console handler and set the level to INFO
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(_formatter)

    # Add the handlers to the logger
    _logger.addHandler(_file_handler)
    _logger.addHandler(_console_handler)

def get_logger():
    assert _logger is not None, "Logger is not initialized. Please call init_logger() first."
    return _logger