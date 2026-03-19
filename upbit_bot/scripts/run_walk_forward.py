"""
scripts/run_walk_forward.py — Walk-Forward 백테스트 실행 스크립트

사용법:
    python upbit_bot/scripts/run_walk_forward.py \
        --coins BTC,ETH,XRP \
        --start 2022-01-01 \
        --end   2024-12-31 \
        --output-dir results/wf

동작:
  1. LookaheadBiasChecker 자동 실행 (오염 피처 발견 시 즉시 종료)
  2. WalkForwardOptimizer.run() 실행 (6개월 IS → 1개월 OOS 반복)
  3. 결과 JSON + SQLite 동시 저장
  4. 완료 시 텔레그램 알림 전송
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import logging.handlers
import sqlite3
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────
# 경로 설정 (coin_history_collector.py와 동일 패턴)
# ─────────────────────────────────────────────────────────────────

_SCRIPT_DIR  = Path(__file__).parent          # upbit_bot/scripts/
_BOT_DIR     = _SCRIPT_DIR.parent             # upbit_bot/
_PROJECT_DIR = _BOT_DIR.parent                # ~/quant/

if str(_BOT_DIR) not in sys.path:
    sys.path.insert(0, str(_BOT_DIR))

LOG_DIR      = _PROJECT_DIR / "logs"
DEFAULT_DB   = _BOT_DIR / "data" / "bot.db"
CONFIG_PATH  = _BOT_DIR / "config.yaml"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 로깅
# ─────────────────────────────────────────────────────────────────

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
))

_file_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / "run_walk_forward.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
))

logging.basicConfig(level=logging.INFO, handlers=[_console, _file_handler])
logger = logging.getLogger("run_walk_forward")

# ─────────────────────────────────────────────────────────────────
# 지연 import (sys.path 설정 후)
# ─────────────────────────────────────────────────────────────────

def _import_backtest_modules() -> tuple[Any, Any, Any, Any]:
    """backtest 모듈 지연 import. 테스트에서 mock 대체 가능."""
    from backtest.walk_forward import (  # type: ignore[import]
        BacktestEngine,
        BacktestParams,
        SurvivourshipHandler,
        WalkForwardOptimizer,
        WalkForwardResult,
    )
    from backtest.lookahead import LookaheadBiasChecker  # type: ignore[import]
    return (
        WalkForwardOptimizer,
        BacktestEngine,
        SurvivourshipHandler,
        LookaheadBiasChecker,
    )


# ─────────────────────────────────────────────────────────────────
# SQLite 결과 저장 스키마
# ─────────────────────────────────────────────────────────────────

_CREATE_WF_TABLE = """
CREATE TABLE IF NOT EXISTS wf_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    run_at        TEXT    NOT NULL,
    coins         TEXT    NOT NULL,
    data_start    TEXT    NOT NULL,
    data_end      TEXT    NOT NULL,
    cycle_idx     INTEGER NOT NULL,
    is_start      TEXT    NOT NULL,
    is_end        TEXT    NOT NULL,
    oos_start     TEXT    NOT NULL,
    oos_end       TEXT    NOT NULL,
    is_sharpe     REAL    NOT NULL,
    oos_sharpe    REAL    NOT NULL,
    overfitting   INTEGER NOT NULL,
    lookahead_ok  INTEGER NOT NULL,
    n_oos_trades  INTEGER NOT NULL
);
"""

_CREATE_WF_SUMMARY = """
CREATE TABLE IF NOT EXISTS wf_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL UNIQUE,
    run_at          TEXT    NOT NULL,
    coins           TEXT    NOT NULL,
    data_start      TEXT    NOT NULL,
    data_end        TEXT    NOT NULL,
    n_cycles        INTEGER NOT NULL,
    avg_oos_sharpe  REAL    NOT NULL,
    avg_is_sharpe   REAL    NOT NULL,
    overfitting_cnt INTEGER NOT NULL,
    live_ready      INTEGER NOT NULL,
    failures_json   TEXT    NOT NULL
);
"""


# ─────────────────────────────────────────────────────────────────
# argparse
# ─────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 인수 파싱."""
    parser = argparse.ArgumentParser(
        prog="run_walk_forward",
        description="Walk-Forward 백테스트 실행 (Lookahead Bias 자동 검증 포함)",
    )
    parser.add_argument(
        "--coins",
        required=True,
        help="쉼표 구분 코인 목록. 예: BTC,ETH,XRP",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="백테스트 시작일 (YYYY-MM-DD). 예: 2022-01-01",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="백테스트 종료일 (YYYY-MM-DD). 예: 2024-12-31",
    )
    parser.add_argument(
        "--output-dir",
        default="results/walk_forward",
        help="결과 저장 디렉터리 (기본: results/walk_forward)",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB),
        help=f"SQLite DB 경로 (기본: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--is-months",
        type=int,
        default=6,
        help="Walk-Forward IS(학습) 구간 개월 수 (기본: 6)",
    )
    parser.add_argument(
        "--oos-months",
        type=int,
        default=1,
        help="Walk-Forward OOS(테스트) 구간 개월 수 (기본: 1)",
    )
    parser.add_argument(
        "--skip-lookahead",
        action="store_true",
        help="Lookahead Bias 검증 건너뜀 (디버그용, 운영환경 사용 금지)",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="텔레그램 알림 비활성화",
    )
    return parser.parse_args(argv)


# ─────────────────────────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────────────────────────

def load_ohlcv_data(
    coins: list[str],
    start: str,
    end: str,
    db_path: str,
) -> "Any":  # pd.DataFrame
    """SQLite 캐시에서 OHLCV + 피처 데이터 로드.

    Phase B 이전에는 데이터가 없으므로 RuntimeError 발생.
    데이터가 준비되면 data/cache.py의 CandleCache를 통해 로드하도록 구현할 것.

    Returns:
        DatetimeIndex를 가진 피처 DataFrame

    Raises:
        RuntimeError: 데이터 미확보 시
    """
    try:
        import pandas as pd
        from data.cache import CandleCache  # type: ignore[import]

        cache = CandleCache(db_path=db_path)
        frames: list[pd.DataFrame] = []

        for coin in coins:
            df = cache.load(market=f"KRW-{coin}", start=start, end=end)
            if df is not None and not df.empty:
                df["coin"] = coin
                frames.append(df)

        if not frames:
            raise RuntimeError(
                f"데이터 없음 — coins={coins} start={start} end={end}\n"
                "Phase B 이전에는 coin_history_collector.py로 최소 6개월 데이터를 수집하라."
            )

        combined = pd.concat(frames, ignore_index=False).sort_index()
        logger.info(
            "[run_wf] 데이터 로드 완료: %d행, 코인=%s, 기간=%s~%s",
            len(combined), coins, start, end,
        )
        return combined

    except ImportError as exc:
        raise RuntimeError(
            f"data.cache 모듈 로드 실패: {exc}\n"
            "upbit_bot 디렉터리를 sys.path에 추가했는지 확인하라."
        ) from exc


# ─────────────────────────────────────────────────────────────────
# Lookahead Bias 사전 검증
# ─────────────────────────────────────────────────────────────────

def run_lookahead_check(df: "Any", LookaheadBiasChecker: type) -> None:
    """LookaheadBiasChecker 실행. 오염 피처 발견 시 ValueError 발생.

    Args:
        df: 피처 DataFrame
        LookaheadBiasChecker: 클래스 (테스트에서 mock 주입 가능)

    Raises:
        ValueError: Lookahead Bias 오염 피처 발견 시
    """
    checker = LookaheadBiasChecker()
    # 전체 인덱스 대신 100개 샘플로 검증 (성능)
    signal_ts = list(df.index[:100]) if hasattr(df, "index") else []
    report = checker.check(df, signal_ts)

    logger.info("[run_wf] %s", report.summary())

    if not report.passed:
        raise ValueError(
            f"Lookahead Bias 오염 피처 {report.contamination_count}개 발견 — "
            f"백테스트 중단: {report.contaminated_features}"
        )


# ─────────────────────────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────────────────────────

def save_json(
    result: "Any",
    output_dir: Path,
    run_id: str,
    coins: list[str],
    start: str,
    end: str,
    failures: list[str],
) -> Path:
    """Walk-Forward 결과를 JSON으로 저장.

    Returns:
        저장된 파일 경로
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"wf_{run_id}.json"

    cycles_data = []
    for c in result.cycles:
        cycle_dict: dict[str, Any] = {
            "cycle_idx":      c.cycle_idx,
            "is_start":       c.is_start,
            "is_end":         c.is_end,
            "oos_start":      c.oos_start,
            "oos_end":        c.oos_end,
            "is_sharpe":      round(c.is_sharpe, 4),
            "oos_sharpe":     round(c.oos_sharpe, 4),
            "is_oos_ratio":   round(c.is_oos_ratio, 4),
            "overfitting":    c.overfitting_flag,
            "lookahead_ok":   c.lookahead_passed,
            "n_oos_trades":   c.n_oos_trades,
        }
        cycles_data.append(cycle_dict)

    payload: dict[str, Any] = {
        "run_id":           run_id,
        "run_at":           datetime.now(timezone.utc).isoformat(),
        "coins":            coins,
        "data_start":       start,
        "data_end":         end,
        "avg_oos_sharpe":   round(result.avg_oos_sharpe, 4),
        "avg_is_sharpe":    round(result.avg_is_sharpe, 4),
        "n_cycles":         len(result.cycles),
        "overfitting_cnt":  result.overfitting_cycles,
        "live_ready":       len(failures) == 0,
        "failures":         failures,
        "cycles":           cycles_data,
    }

    filename.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[run_wf] JSON 저장: %s", filename)
    return filename


def save_sqlite(
    result: "Any",
    db_path: str,
    run_id: str,
    coins: list[str],
    start: str,
    end: str,
    failures: list[str],
) -> None:
    """Walk-Forward 결과를 SQLite에 저장 (wf_results + wf_summary 테이블)."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_CREATE_WF_TABLE)
        conn.execute(_CREATE_WF_SUMMARY)

        run_at = datetime.now(timezone.utc).isoformat()
        coins_str = ",".join(coins)

        rows = [
            (
                run_id, run_at, coins_str, start, end,
                c.cycle_idx,
                c.is_start, c.is_end,
                c.oos_start, c.oos_end,
                round(c.is_sharpe, 6),
                round(c.oos_sharpe, 6),
                int(c.overfitting_flag),
                int(c.lookahead_passed),
                c.n_oos_trades,
            )
            for c in result.cycles
        ]

        conn.executemany(
            """
            INSERT INTO wf_results
              (run_id, run_at, coins, data_start, data_end,
               cycle_idx, is_start, is_end, oos_start, oos_end,
               is_sharpe, oos_sharpe, overfitting, lookahead_ok, n_oos_trades)
            VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?)
            """,
            rows,
        )

        conn.execute(
            """
            INSERT OR REPLACE INTO wf_summary
              (run_id, run_at, coins, data_start, data_end,
               n_cycles, avg_oos_sharpe, avg_is_sharpe, overfitting_cnt,
               live_ready, failures_json)
            VALUES (?,?,?,?,?, ?,?,?,?,?,?)
            """,
            (
                run_id, run_at, coins_str, start, end,
                len(result.cycles),
                round(result.avg_oos_sharpe, 6),
                round(result.avg_is_sharpe, 6),
                result.overfitting_cycles,
                int(len(failures) == 0),
                json.dumps(failures, ensure_ascii=False),
            ),
        )

        conn.commit()
        logger.info("[run_wf] SQLite 저장 완료: %s (사이클 %d개)", db_path, len(result.cycles))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────
# 텔레그램 알림
# ─────────────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    """텔레그램 one-shot 알림 전송.

    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID 환경변수 필요.
    미설정 시 경고 로그만 출력하고 계속 진행.

    Returns:
        True: 전송 성공, False: 전송 실패 (환경변수 미설정·라이브러리 미설치·네트워크 오류 포함)
    """
    import os

    token   = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        logger.warning("[run_wf] TELEGRAM_TOKEN / TELEGRAM_CHAT_ID 미설정 — 알림 건너뜀")
        return False

    try:
        from telegram import Bot  # type: ignore[import]

        async def _send() -> None:
            async with Bot(token=token) as bot:
                await bot.send_message(
                    chat_id=int(chat_id),
                    text=message,
                    parse_mode="HTML",
                )

        asyncio.run(_send())
        logger.info("[run_wf] 텔레그램 알림 전송 완료")
        return True
    except ImportError:
        logger.warning("[run_wf] python-telegram-bot 미설치 — 텔레그램 알림 건너뜀")
        return False
    except Exception as exc:
        # 알림 실패는 치명적이지 않으나 인지할 수 있도록 warning 출력 후 False 반환
        logger.warning("[run_wf] 텔레그램 알림 실패: %s", exc)
        return False


def _build_telegram_message(
    result: "Any",
    run_id: str,
    coins: list[str],
    start: str,
    end: str,
    failures: list[str],
    elapsed_sec: float,
) -> str:
    """텔레그램 결과 메시지 포맷."""
    status = "✅ 실전 전환 가능" if not failures else f"⚠️ 미통과 {len(failures)}개"
    lines = [
        "<b>📊 Walk-Forward 백테스트 완료</b>",
        f"Run ID: <code>{run_id}</code>",
        f"코인: {', '.join(coins)}",
        f"기간: {start} ~ {end}",
        f"사이클: {len(result.cycles)}개",
        f"평균 OOS 샤프: {result.avg_oos_sharpe:.3f}",
        f"평균 IS 샤프:  {result.avg_is_sharpe:.3f}",
        f"과적합 사이클: {result.overfitting_cycles}/{len(result.cycles)}",
        f"실전 전환 상태: {status}",
        f"소요시간: {elapsed_sec:.0f}초",
    ]
    if failures:
        lines.append("\n<b>미통과 기준:</b>")
        for f in failures:
            lines.append(f"  • {f}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Walk-Forward 백테스트 스크립트 진입점.

    Returns:
        0 = 정상 완료, 1 = 오류
    """
    import time

    args = parse_args(argv)

    coins     = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    start     = args.start
    end       = args.end
    output_dir = Path(args.output_dir)
    db_path   = args.db_path

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info(
        "[run_wf] 시작 — run_id=%s coins=%s start=%s end=%s output=%s",
        run_id, coins, start, end, output_dir,
    )

    t_start = time.monotonic()

    # ── 모듈 import ──
    (
        WalkForwardOptimizer,
        BacktestEngine,
        SurvivourshipHandler,
        LookaheadBiasChecker,
    ) = _import_backtest_modules()

    # ── 데이터 로딩 ──
    try:
        df = load_ohlcv_data(coins=coins, start=start, end=end, db_path=db_path)
    except RuntimeError as exc:
        logger.error("[run_wf] 데이터 로딩 실패: %s", exc)
        return 1

    # ── Lookahead Bias 사전 검증 ──
    if not args.skip_lookahead:
        try:
            run_lookahead_check(df, LookaheadBiasChecker)
        except ValueError as exc:
            logger.error("[run_wf] %s", exc)
            return 1
    else:
        logger.warning("[run_wf] --skip-lookahead 활성화 — Lookahead 검증 생략 (운영환경 금지)")

    # ── Walk-Forward 실행 ──
    survivourship = SurvivourshipHandler()
    loaded_history = survivourship.load_from_db(db_path)
    if not loaded_history:
        raise ValueError(
            "coin_history 데이터 없음 — "
            "scripts/coin_history_collector.py 먼저 실행 필요"
        )

    engine    = BacktestEngine(survivourship=survivourship)
    optimizer = WalkForwardOptimizer(
        engine=engine,
        is_months=args.is_months,
        oos_months=args.oos_months,
    )

    from backtest.walk_forward import BacktestParams  # type: ignore[import]

    def default_optimize_fn(df_is: "Any") -> "Any":
        """IS 구간 최적화 함수 — 기본 파라미터 반환 (Hyperopt 미실행 시)."""
        return BacktestParams()

    logger.info("[run_wf] Walk-Forward 실행 중...")
    result = optimizer.run(df=df, optimize_fn=default_optimize_fn)

    # 실전 전환 기준 체크
    failures = optimizer.check_live_readiness(result)

    elapsed = time.monotonic() - t_start
    logger.info("[run_wf] %s\n소요시간: %.1f초", result.summary(), elapsed)

    if failures:
        logger.warning("[run_wf] 실전 전환 미통과: %s", failures)
    else:
        logger.info("[run_wf] 실전 전환 7가지 기준 모두 통과")

    # ── 결과 저장 ──
    json_path = save_json(
        result=result,
        output_dir=output_dir,
        run_id=run_id,
        coins=coins,
        start=start,
        end=end,
        failures=failures,
    )
    logger.info("[run_wf] JSON 경로: %s", json_path)

    save_sqlite(
        result=result,
        db_path=db_path,
        run_id=run_id,
        coins=coins,
        start=start,
        end=end,
        failures=failures,
    )

    # ── 텔레그램 알림 ──
    if not args.no_telegram:
        msg = _build_telegram_message(
            result=result,
            run_id=run_id,
            coins=coins,
            start=start,
            end=end,
            failures=failures,
            elapsed_sec=elapsed,
        )
        if not send_telegram(msg):
            logger.warning("[run_wf] 텔레그램 알림 미전송 — run_id=%s", run_id)

    logger.info("[run_wf] 완료 — run_id=%s", run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
