"""统一日志。"""

from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED = False
_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
    "<cyan>{extra[tag]: <10}</cyan> | <level>{message}</level>"
)


def setup_logger(level: str = "INFO"):
    global _CONFIGURED
    logger.remove()
    logger.configure(extra={"tag": "app"})
    logger.add(sys.stderr, level=(level or "INFO").upper(), format=_FORMAT, colorize=True, backtrace=False)
    _CONFIGURED = True
    return logger


def get_logger(tag: str = "app"):
    if not _CONFIGURED:
        setup_logger()
    return logger.bind(tag=tag)


def banner(text: str, char: str = "=") -> str:
    line = char * max(len(text) + 4, 40)
    return f"\n{line}\n  {text}\n{line}"
