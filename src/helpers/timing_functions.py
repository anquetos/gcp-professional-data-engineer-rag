"""This module contains a decorator that can be used to time the execution of a function."""

import logging
import time
from functools import wraps

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def timing(func):
    """
    A decorator that prints the execution time of the decorated function.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The wrapped function with timing functionality.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"'{func.__name__}' function executed in {duration:.5f} seconds.")
        return result

    return wrapper
