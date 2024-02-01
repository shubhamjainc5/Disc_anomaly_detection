"""class and methods for logs handling."""

import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s', level=logging.DEBUG,
    handlers=[
        RotatingFileHandler("logs/app.log", mode='a',maxBytes=1048576, backupCount=20, encoding ='utf'),
        logging.StreamHandler()
    ])

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

Logger = logging.getLogger('root')


# class Logger():
#     """class def handling logs."""

#     @staticmethod
#     def info(message):
#         """Display info logs."""
#         logging.info(message)

#     @staticmethod
#     def warning(message):
#         """Display warning logs."""
#         logging.warning(message)

#     @staticmethod
#     def debug(message):
#         """Display debug logs."""
#         logging.debug(message)

#     @staticmethod
#     def error(message):
#         """Display error logs."""
#         logging.error(message)