"""class and methods for logs handling."""

import logging

logging.basicConfig(filename='app.log',
                    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', level=logging.DEBUG)

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

class Logger():
    """class def handling logs."""

    @staticmethod
    def info(message):
        """Display info logs."""
        logging.info(message)

    @staticmethod
    def warning(message):
        """Display warning logs."""
        logging.warning(message)

    @staticmethod
    def debug(message):
        """Display debug logs."""
        logging.debug(message)

    @staticmethod
    def error(message):
        """Display error logs."""
        logging.error(message)