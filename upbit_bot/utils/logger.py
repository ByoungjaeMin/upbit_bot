"""
utils/logger.py — 봇 전용 로거 팩토리

표준 logging 위에 아래 기능을 추가:
  - RotatingFileHandler (10MB × 5개)
  - UTC 타임스탬프 고정
  - 거래 전용 핸들러 (trade 이름 하위 로거만 별도 파일 저장)

사용법:
    from utils.logger import get_logger, setup_root_logger

    setup_root_logger(log_dir="/path/to/logs", level=logging.INFO)
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


# 로그 포맷 — UTC 고정
_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"

# 로그 파일 크기 설정
_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
_BACKUP_COUNT = 5


class _UTCFormatter(logging.Formatter):
    """UTC 타임스탬프를 사용하는 포맷터."""
    converter = __import__("time").gmtime


def setup_root_logger(
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    console: bool = True,
) -> None:
    """루트 로거 초기화.

    log_dir 아래 bot.log (RotatingFile) + 선택적 콘솔 핸들러를 추가.
    중복 호출 시 기존 핸들러를 교체하지 않고 무시.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        return  # 이미 설정됨

    root.setLevel(level)
    formatter = _UTCFormatter(fmt=_FMT, datefmt=_DATEFMT)

    # RotatingFileHandler — bot.log
    fh = logging.handlers.RotatingFileHandler(
        log_path / "bot.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # 거래 전용 핸들러 — trade.log
    trade_logger = logging.getLogger("trade")
    trade_logger.setLevel(logging.DEBUG)
    if not trade_logger.handlers:
        tfh = logging.handlers.RotatingFileHandler(
            log_path / "trade.log",
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        tfh.setFormatter(formatter)
        trade_logger.addHandler(tfh)

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        root.addHandler(ch)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """이름 기반 로거 반환. level 미지정 시 부모 레벨 상속."""
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def get_trade_logger() -> logging.Logger:
    """trade.log 전용 로거 반환."""
    return logging.getLogger("trade")
