"""
scripts/coin_history_collector.py — 업비트 KRW 마켓 코인 목록 일일 스냅샷

독립 실행 스크립트 (main.py와 무관).

  python upbit_bot/scripts/coin_history_collector.py          # APScheduler 데몬
  python upbit_bot/scripts/coin_history_collector.py --now    # 즉시 1회 실행 후 종료

동작:
  1. 매일 00:05 UTC — pyupbit.get_tickers("KRW") 조회
  2. KRW 마켓 전체 + 24시간 거래량(원화) + 거래량 순위 계산
  3. coin_history 테이블 UPSERT (snapshot_date, coin) UNIQUE 충돌 무시
  4. 저장 완료 → 콘솔 출력 (날짜 + 코인 수)
  5. 오류 → logs/coin_history_error.log 기록

DB: upbit_bot/data/bot.db (coin_history 테이블은 schema.py에 DDL 정의됨)
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────

_SCRIPT_DIR  = Path(__file__).parent          # upbit_bot/scripts/
_BOT_DIR     = _SCRIPT_DIR.parent             # upbit_bot/
_PROJECT_DIR = _BOT_DIR.parent                # ~/quant/

DB_PATH      = _BOT_DIR / "data" / "bot.db"
LOG_DIR      = _PROJECT_DIR / "logs"
ERROR_LOG    = LOG_DIR / "coin_history_error.log"

# ─────────────────────────────────────────────────────────────────
# 로깅
# ─────────────────────────────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                        datefmt="%Y-%m-%dT%H:%M:%SZ"))

_file_handler = logging.handlers.RotatingFileHandler(
    ERROR_LOG, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setLevel(logging.WARNING)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ"
))

logger = logging.getLogger("coin_history")
logger.setLevel(logging.INFO)
logger.addHandler(_console)
logger.addHandler(_file_handler)


# ─────────────────────────────────────────────────────────────────
# 핵심 수집 함수
# ─────────────────────────────────────────────────────────────────

def collect_and_save() -> int:
    """KRW 마켓 코인 목록 수집 → coin_history 저장.

    Returns:
        저장된 코인 수. 실패 시 예외 전파.
    """
    try:
        import pyupbit
    except ImportError as e:
        raise RuntimeError("pyupbit 미설치 — pip install pyupbit") from e

    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info("수집 시작 snapshot_date=%s", snapshot_date)

    # 1. KRW 마켓 전체 티커 목록
    tickers: list[str] = pyupbit.get_tickers(fiat="KRW")
    if not tickers:
        raise RuntimeError("get_tickers 결과 없음 — API 오류 또는 네트워크 문제")

    logger.info("티커 조회 완료: %d개", len(tickers))

    # 2. 24시간 현재가 + 거래량 조회 (최대 100개씩 분할 요청)
    volume_map: dict[str, float] = {}

    chunk_size = 100
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        try:
            current_prices = pyupbit.get_current_price(chunk)
            if isinstance(current_prices, dict):
                # get_current_price는 가격만 반환 — 거래량은 별도 조회
                pass
        except Exception as exc:
            logger.warning("현재가 조회 오류 (chunk %d): %s", i // chunk_size, exc)

        # 거래량 조회: get_ohlcv로 최신 1일봉에서 volume 추출
        for ticker in chunk:
            try:
                df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
                if df is not None and not df.empty:
                    close  = float(df["close"].iloc[-1])
                    volume = float(df["volume"].iloc[-1])
                    volume_map[ticker] = close * volume  # KRW 거래량
            except Exception:
                volume_map.setdefault(ticker, 0.0)

        # API 레이트 리밋 방어 (100개 청크당 0.5초 대기)
        if i + chunk_size < len(tickers):
            time.sleep(0.5)

    # 3. 거래량 기준 순위 계산
    sorted_tickers = sorted(
        tickers,
        key=lambda t: volume_map.get(t, 0.0),
        reverse=True,
    )
    rank_map = {ticker: rank for rank, ticker in enumerate(sorted_tickers, start=1)}

    # 4. SQLite UPSERT
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        rows = [
            (
                snapshot_date,
                ticker,
                round(volume_map.get(ticker, 0.0), 0),
                rank_map[ticker],
                None,   # market_cap_krw — 공개 API 미제공
                0,      # included_in_pairlist (collector.py가 별도 갱신)
            )
            for ticker in tickers
        ]

        conn.executemany(
            """INSERT OR IGNORE INTO coin_history
               (snapshot_date, coin, volume_24h_krw, rank, market_cap_krw, included_in_pairlist)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        saved = conn.execute(
            "SELECT COUNT(*) FROM coin_history WHERE snapshot_date = ?",
            (snapshot_date,),
        ).fetchone()[0]
    finally:
        conn.close()

    logger.info(
        "저장 완료 | 날짜: %s | 코인 수: %d개 | DB: %s",
        snapshot_date, saved, DB_PATH,
    )
    # 콘솔 강조 출력
    print(f"\n{'='*50}")
    print(f"  coin_history 저장 완료")
    print(f"  날짜  : {snapshot_date}")
    print(f"  코인 수: {saved}개")
    print(f"  DB    : {DB_PATH}")
    print(f"{'='*50}\n")

    return saved


# ─────────────────────────────────────────────────────────────────
# APScheduler 데몬 모드
# ─────────────────────────────────────────────────────────────────

def _scheduled_job() -> None:
    """APScheduler에서 호출되는 래퍼 — 예외를 로그로 흡수."""
    try:
        collect_and_save()
    except Exception as exc:
        logger.error("수집 실패: %s", exc, exc_info=True)


def run_scheduler() -> None:
    """매일 00:05 UTC 자동 실행 데몬."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError as e:
        logger.error("apscheduler 미설치 — pip install apscheduler: %s", e)
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        _scheduled_job,
        CronTrigger(hour=0, minute=5),
        id="coin_history_daily",
        name="KRW 코인 히스토리 일일 스냅샷",
        misfire_grace_time=300,      # 5분 내 지연 허용
        max_instances=1,
    )

    job = scheduler.get_jobs()[0]
    next_run = getattr(job, "next_fire_time", None) or getattr(job, "next_run_time", None)
    logger.info("APScheduler 시작 — 다음 실행: %s (매일 00:05 UTC)", next_run)
    logger.info("Ctrl+C로 종료")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("스케줄러 종료")


# ─────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="업비트 KRW 마켓 코인 히스토리 수집기"
    )
    parser.add_argument(
        "--now",
        action="store_true",
        help="즉시 1회 실행 후 종료 (스케줄러 없이)",
    )
    args = parser.parse_args()

    # DB 파일 존재 여부 사전 확인
    if not DB_PATH.exists():
        logger.error(
            "DB 파일 없음: %s\n"
            "  → 먼저 main.py를 한 번 실행해 DB를 초기화하세요.",
            DB_PATH,
        )
        sys.exit(1)

    if args.now:
        try:
            collect_and_save()
        except Exception as exc:
            logger.error("수집 실패: %s", exc, exc_info=True)
            sys.exit(1)
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
