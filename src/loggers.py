import os
import logging
import sys
from typing import Union


def build_logger(
    log_file: Union[os.PathLike, str]
) -> logging.Logger:
    # 创建Logger对象，设置最低日志级别（DEBUG包含所有级别）
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # 创建文件处理器（输出到文件，级别为DEBUG）
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 创建控制台处理器（输出到终端，级别为INFO）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # 将处理器添加到Logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger