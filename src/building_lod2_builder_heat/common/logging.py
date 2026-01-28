import logging
import os
import sys
from enum import Enum
from pathlib import Path

from loguru import logger


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class InterceptHandler(logging.Handler):
    """
    標準のロギング出力を loguru に転送するハンドラ。
    """

    _LEVEL_MAP = {
        "CRITICAL": "ERROR",
        "ERROR": "WARNING",
        "WARNING": "INFO",
        "INFO": "DEBUG",
        "DEBUG": "TRACE",
    }
    _logging_file = os.path.normcase(logging.__file__)

    def emit(self, record):
        # 対応する loguru のレベルを取得
        level = self._LEVEL_MAP.get(record.levelname, record.levelname)

        # 呼び出し元の情報を特定
        frame, depth = logging.currentframe().f_back, 2
        while frame:
            filename = os.path.normcase(frame.f_code.co_filename)
            # logging モジュール内部、またはこのハンドラ自身の中にいる間はさかのぼる
            if filename == self._logging_file or filename == os.path.normcase(__file__):
                frame = frame.f_back
                depth += 1
            else:
                break

        logger.opt(depth=depth - 1, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger(level: LogLevel, log_file: Path | None = None):
    """
    ロガーの初期化設定。

    :param level: 標準出力に出力するログレベル。
    :param log_file: ログファイルを出力する場合のパス。
    """
    # 既存のハンドラを削除（デフォルトの stderr 出力をリセット）
    logger.remove()

    # 標準出力（コンソール）への出力設定
    logger.add(sys.stderr, level=level.value)

    # ファイルへの出力設定（指定がある場合）
    if log_file:
        logger.add(log_file, level="DEBUG", rotation="10 MB")

    # 標準の logging モジュールからの出力を loguru でキャプチャする
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
