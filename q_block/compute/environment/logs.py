"""Centralised logging configuration for qBlock.

Call :func:`setup_logging` once at the top of any entry-point script::

    from q_block.compute.environment.logs import setup_logging

    logger = setup_logging(__name__)
    logger.info("Ready.")

Library modules should **not** call this function; they should only
create their own loggers with ``logging.getLogger(__name__)``.
"""

import logging
from typing import Optional

DEFAULT_FORMAT = "%(name)s %(levelname)s: %(message)s"


def setup_logging(
    name: Optional[str] = None,
    level: int = logging.INFO,
    fmt: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """Configure the root logger and return a named logger.

    :param name: Logger name (typically ``__name__``).  When *None*
        the root logger is returned.
    :type name: Optional[str]
    :param level: Logging level applied to the root logger.
    :type level: int
    :param fmt: Format string for log records.
    :type fmt: str
    :returns: A logger instance ready to use.
    :rtype: logging.Logger
    """
    logging.basicConfig(level=level, format=fmt)
    return logging.getLogger(name)
