"""
scripts/run_hyperopt.py — Optuna Hyperopt 실행 스크립트

사용법:
    python upbit_bot/scripts/run_hyperopt.py \
        --n-trials 200 \
        --strategy all \
        --output-dir results/hyperopt

동작:
  1. HyperoptEngine.optimize_with_walkforward() 실행
     목적함수: Walk-Forward OOS 샤프비율 최대화
  2. 최적 파라미터 config.yaml의 backtest_params 섹션 자동 업데이트
  3. 결과 JSON 저장
  4. 완료 시 텔레그램 알림 전송
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────

_SCRIPT_DIR  = Path(__file__).parent          # upbit_bot/scripts/
_BOT_DIR     = _SCRIPT_DIR.parent             # upbit_bot/
_PROJECT_DIR = _BOT_DIR.parent                # ~/quant/

if str(_BOT_DIR) not in sys.path:
    sys.path.insert(0, str(_BOT_DIR))

LOG_DIR     = _PROJECT_DIR / "logs"
DEFAULT_DB  = _BOT_DIR / "data" / "bot.db"
CONFIG_PATH = _BOT_DIR / "config.yaml"

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
    LOG_DIR / "run_hyperopt.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
))

logging.basicConfig(level=logging.INFO, handlers=[_console, _file_handler])
logger = logging.getLogger("run_hyperopt")

# ─────────────────────────────────────────────────────────────────
# 지원 전략 목록
# ─────────────────────────────────────────────────────────────────

VALID_STRATEGIES = ("all", "trend", "grid", "dca")


# ─────────────────────────────────────────────────────────────────
# 지연 import
# ─────────────────────────────────────────────────────────────────

def _import_hyperopt_modules() -> tuple[Any, Any]:
    """backtest.hyperopt 모듈 지연 import. 테스트에서 mock 대체 가능."""
    from backtest.hyperopt import HyperoptEngine  # type: ignore[import]
    from backtest.walk_forward import BacktestEngine  # type: ignore[import]
    return HyperoptEngine, BacktestEngine


# ─────────────────────────────────────────────────────────────────
# argparse
# ─────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 인수 파싱."""
    parser = argparse.ArgumentParser(
        prog="run_hyperopt",
        description="Optuna Walk-Forward Hyperopt 실행 (OOS 샤프비율 최대화)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Optuna trial 횟수 (기본: 200, M4에서 약 1~2시간)",
    )
    parser.add_argument(
        "--strategy",
        choices=VALID_STRATEGIES,
        default="all",
        help=(
            "최적화 대상 전략 (기본: all). "
            "all=전체, trend=추세추종, grid=그리드, dca=DCA"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/hyperopt",
        help="결과 저장 디렉터리 (기본: results/hyperopt)",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB),
        help=f"SQLite DB 경로 (기본: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--config-path",
        default=str(CONFIG_PATH),
        help=f"config.yaml 경로 (기본: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--coins",
        default="BTC,ETH,XRP",
        help="학습 데이터 코인 (기본: BTC,ETH,XRP)",
    )
    parser.add_argument(
        "--start",
        default="2022-01-01",
        help="학습 데이터 시작일 (기본: 2022-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2024-12-31",
        help="학습 데이터 종료일 (기본: 2024-12-31)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Optuna 타임아웃 초 (기본: 7200 = 2시간)",
    )
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="config.yaml 자동 업데이트 비활성화",
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
            "[run_hyperopt] 데이터 로드: %d행, 코인=%s, 기간=%s~%s",
            len(combined), coins, start, end,
        )
        return combined

    except ImportError as exc:
        raise RuntimeError(
            f"data.cache 모듈 로드 실패: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────
# 전략별 strategy_fn 선택
# ─────────────────────────────────────────────────────────────────

def _get_strategy_fn(strategy: str) -> "Any | None":
    """전략 이름에서 backtest strategy_fn 반환.

    현재는 all/trend/grid/dca 모두 None 반환
    (BacktestEngine 기본 룰 기반 신호 사용).
    Phase C 이후 전략별 신호 함수로 교체 예정.

    Args:
        strategy: 'all' | 'trend' | 'grid' | 'dca'

    Returns:
        strategy_fn or None
    """
    # Phase C 이후 전략별 callable 로 교체:
    # if strategy == "trend":
    #     from strategies.trend import TrendStrategy
    #     return TrendStrategy().generate_signals
    # elif strategy == "grid":
    #     from strategies.grid import GridStrategy
    #     return GridStrategy().generate_signals
    # elif strategy == "dca":
    #     from strategies.dca import DCAStrategy
    #     return DCAStrategy().generate_signals
    return None  # 기본 룰 기반 신호 사용 (BacktestEngine._default_rule_based)


# ─────────────────────────────────────────────────────────────────
# config.yaml 업데이트
# ─────────────────────────────────────────────────────────────────

def update_config_yaml(
    best_params: "Any",  # BacktestParams
    config_path: str | Path,
    strategy: str = "all",
) -> None:
    """최적 파라미터를 config.yaml의 backtest_params 섹션에 저장.

    기존 config.yaml의 다른 섹션은 그대로 보존.
    backtest_params 섹션만 upsert.

    Args:
        best_params: BacktestParams 인스턴스
        config_path: config.yaml 경로
        strategy: 업데이트 대상 전략 ('all' | 'trend' | 'grid' | 'dca')

    Raises:
        ImportError: PyYAML 미설치 시
        OSError: 파일 읽기/쓰기 실패 시
    """
    import yaml  # type: ignore[import]

    config_path = Path(config_path)

    # 기존 설정 로드
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        logger.warning("[run_hyperopt] config.yaml 없음 — 새 파일 생성: %s", config_path)
        config = {}

    # backtest_params 섹션 upsert
    if "backtest_params" not in config:
        config["backtest_params"] = {}

    section_key = "all" if strategy == "all" else strategy

    config["backtest_params"][section_key] = {
        "rsi_period":           best_params.rsi_period,
        "ema_short":            best_params.ema_short,
        "ema_long":             best_params.ema_long,
        "adx_threshold":        round(best_params.adx_threshold, 2),
        "stop_loss_pct":        round(best_params.stop_loss_pct, 4),
        "take_profit_pct":      round(best_params.take_profit_pct, 4),
        "kelly_fraction":       round(best_params.kelly_fraction, 4),
        "atr_multiplier":       round(best_params.atr_multiplier, 2),
        "grid_levels":          best_params.grid_levels,
        "dca_step_pct":         round(best_params.dca_step_pct, 4),
        "ensemble_threshold":   round(best_params.ensemble_threshold, 4),
    }
    config["backtest_params"]["_last_updated"] = datetime.now(timezone.utc).isoformat()
    config["backtest_params"]["_strategy"]     = strategy

    # 저장 (PyYAML — 주석은 유지되지 않음, 값만 업데이트)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info(
        "[run_hyperopt] config.yaml 업데이트 완료: backtest_params[%s] → %s",
        section_key,
        config_path,
    )


# ─────────────────────────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────────────────────────

def save_json(
    result: "Any",
    output_dir: Path,
    run_id: str,
    strategy: str,
    n_trials: int,
) -> Path:
    """Hyperopt 결과를 JSON으로 저장.

    Returns:
        저장된 파일 경로
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"hyperopt_{strategy}_{run_id}.json"

    best = result.best_params
    payload: dict[str, Any] = {
        "run_id":              run_id,
        "run_at":              datetime.now(timezone.utc).isoformat(),
        "strategy":            strategy,
        "n_trials_requested":  n_trials,
        "n_trials_completed":  result.n_trials_completed,
        "study_name":          result.study_name,
        "best_oos_sharpe":     round(result.best_oos_sharpe, 4),
        "best_params": {
            "rsi_period":         best.rsi_period,
            "ema_short":          best.ema_short,
            "ema_long":           best.ema_long,
            "adx_threshold":      round(best.adx_threshold, 2),
            "stop_loss_pct":      round(best.stop_loss_pct, 4),
            "take_profit_pct":    round(best.take_profit_pct, 4),
            "kelly_fraction":     round(best.kelly_fraction, 4),
            "atr_multiplier":     round(best.atr_multiplier, 2),
            "grid_levels":        best.grid_levels,
            "dca_step_pct":       round(best.dca_step_pct, 4),
            "ensemble_threshold": round(best.ensemble_threshold, 4),
        },
        "param_importances": result.param_importances,
    }

    filename.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[run_hyperopt] JSON 저장: %s", filename)
    return filename


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
        logger.warning("[run_hyperopt] TELEGRAM_TOKEN / TELEGRAM_CHAT_ID 미설정 — 알림 건너뜀")
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
        logger.info("[run_hyperopt] 텔레그램 알림 전송 완료")
        return True
    except ImportError:
        logger.warning("[run_hyperopt] python-telegram-bot 미설치 — 텔레그램 알림 건너뜀")
        return False
    except Exception as exc:
        # 알림 실패는 치명적이지 않으나 인지할 수 있도록 warning 출력 후 False 반환
        logger.warning("[run_hyperopt] 텔레그램 알림 실패: %s", exc)
        return False


def _build_telegram_message(
    result: "Any",
    run_id: str,
    strategy: str,
    elapsed_sec: float,
    config_updated: bool,
) -> str:
    """텔레그램 결과 메시지 포맷."""
    top_params = sorted(
        result.param_importances.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:3] if result.param_importances else []

    top_str = ", ".join(f"{k}={v:.3f}" for k, v in top_params) if top_params else "N/A"

    lines = [
        "<b>🔬 Hyperopt 최적화 완료</b>",
        f"Run ID: <code>{run_id}</code>",
        f"전략: {strategy}",
        f"최적 OOS 샤프: {result.best_oos_sharpe:.4f}",
        f"완료 trials: {result.n_trials_completed}",
        f"ensemble_threshold: {result.best_params.ensemble_threshold:.3f}",
        f"adx_threshold: {result.best_params.adx_threshold:.1f}",
        f"kelly_fraction: {result.best_params.kelly_fraction:.3f}",
        f"상위 중요 파라미터: {top_str}",
        f"config.yaml 업데이트: {'완료' if config_updated else '건너뜀'}",
        f"소요시간: {elapsed_sec:.0f}초",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Hyperopt 스크립트 진입점.

    Returns:
        0 = 정상 완료, 1 = 오류
    """
    import time

    args = parse_args(argv)

    coins      = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    start      = args.start
    end        = args.end
    strategy   = args.strategy
    n_trials   = args.n_trials
    output_dir = Path(args.output_dir)
    db_path    = args.db_path
    config_path = args.config_path

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info(
        "[run_hyperopt] 시작 — run_id=%s strategy=%s n_trials=%d coins=%s",
        run_id, strategy, n_trials, coins,
    )

    t_start = time.monotonic()

    # ── 모듈 import ──
    HyperoptEngine, BacktestEngine = _import_hyperopt_modules()

    # ── 데이터 로딩 ──
    try:
        df = load_ohlcv_data(coins=coins, start=start, end=end, db_path=db_path)
    except RuntimeError as exc:
        logger.error("[run_hyperopt] 데이터 로딩 실패: %s", exc)
        return 1

    # ── strategy_fn 선택 ──
    strategy_fn = _get_strategy_fn(strategy)

    # ── Hyperopt 실행 ──
    engine = HyperoptEngine(
        backtest_engine=BacktestEngine(),
        n_trials=n_trials,
        timeout_sec=args.timeout,
    )

    logger.info("[run_hyperopt] Walk-Forward Hyperopt 실행 중 (n_trials=%d)...", n_trials)
    # 의도적 비보호: Optuna 내부 오류는 즉시 크래시가 맞음.
    # 조용히 넘어가면 잘못된 파라미터로 config가 업데이트될 위험.
    result = engine.optimize_with_walkforward(
        df_full=df,
        strategy_fn=strategy_fn,
        n_trials=n_trials,
    )

    elapsed = time.monotonic() - t_start
    logger.info("[run_hyperopt] %s\n소요시간: %.1f초", result.summary(), elapsed)

    # ── 결과 저장 ──
    json_path = save_json(
        result=result,
        output_dir=output_dir,
        run_id=run_id,
        strategy=strategy,
        n_trials=n_trials,
    )
    logger.info("[run_hyperopt] JSON 경로: %s", json_path)

    # ── config.yaml 업데이트 ──
    config_updated = False
    if not args.no_update_config:
        update_config_yaml(
            best_params=result.best_params,
            config_path=config_path,
            strategy=strategy,
        )
        config_updated = True

    # ── 텔레그램 알림 ──
    if not args.no_telegram:
        msg = _build_telegram_message(
            result=result,
            run_id=run_id,
            strategy=strategy,
            elapsed_sec=elapsed,
            config_updated=config_updated,
        )
        if not send_telegram(msg):
            logger.warning("[run_hyperopt] 텔레그램 알림 미전송 — run_id=%s", run_id)

    logger.info("[run_hyperopt] 완료 — run_id=%s", run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
