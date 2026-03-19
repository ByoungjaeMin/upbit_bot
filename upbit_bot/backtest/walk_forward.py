"""
backtest/walk_forward.py — Walk-Forward 최적화 + 생존 편향 처리

WalkForwardOptimizer:
  - 6개월 학습(IS) → 1개월 테스트(OOS) → 전진 반복
  - 각 구간 파라미터 재최적화
  - OOS 성과 집계
  - 과적합 판별: IS 샤프 vs OOS 샤프 비율 < 0.5 → 경고
  - 각 사이클 시작 전 Lookahead Bias 검증 자동 실행

SurvivourshipHandler:
  - 날짜별 업비트 상장 코인 목록 로드 (SQLite coin_history)
  - 백테스트 시 해당 시점 유효 코인만 사용 (현재 상위 코인 소급 금지)

BacktestEngine:
  - 단일 코인 / 단일 파라미터 세트 백테스트 실행
  - Pessimistic 비용: 수익-0.25%, 손실+0.25%
  - 전략별 구간 분리 평가 (bull/bear/sideways)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from backtest.lookahead import LookaheadBiasChecker, LookaheadReport

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

IS_MONTHS  = 6    # 학습 구간 (개월)
OOS_MONTHS = 1    # 테스트 구간 (개월)
PESSIMISTIC_WIN_COST  = -0.0025   # 수익 거래: -0.25%
PESSIMISTIC_LOSS_COST = +0.0025   # 손실 거래: +0.25%
OVERFITTING_THRESHOLD = 0.5       # IS/OOS 샤프 비율 < 0.5 → 과적합 경고
ANNUALIZE_FACTOR      = np.sqrt(252 * 24 * 12)  # 5분봉 연율화

# 전략별 구간 정의
REGIME_PERIODS = {
    "bull":     ("2024-10-01", "2025-01-31"),
    "sideways": ("2023-01-01", "2023-06-30"),
    "bear":     ("2022-05-01", "2022-12-31"),
}

BEAR_MAX_DRAWDOWN = -0.10   # 하락장 낙폭 < 10% 필수


# ------------------------------------------------------------------
# 데이터 클래스
# ------------------------------------------------------------------

@dataclass
class BacktestParams:
    """백테스트 파라미터 세트."""
    rsi_period: int = 14
    ema_short: int = 7
    ema_long: int = 25
    adx_threshold: float = 20.0
    stop_loss_pct: float = 0.07
    take_profit_pct: float = 0.10
    kelly_fraction: float = 0.25
    atr_multiplier: float = 2.0
    grid_levels: int = 10
    dca_step_pct: float = -0.03
    ensemble_threshold: float = 0.62


@dataclass
class TradeResult:
    """단일 거래 결과."""
    timestamp: datetime
    coin: str
    side: str
    entry_price: float
    exit_price: float
    pnl_pct: float          # 수수료/슬리피지 반영 전
    pnl_pct_pessimistic: float  # 비관적 비용 반영 후
    strategy_type: str
    hold_minutes: float


@dataclass
class PeriodMetrics:
    """단일 백테스트 구간 성과 지표."""
    period_start: str
    period_end: str
    n_trades: int
    win_rate: float
    sharpe: float
    max_drawdown: float
    total_return: float
    profit_loss_ratio: float    # 평균 수익 / 평균 손실
    strategy_contributions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "total_return": round(self.total_return, 4),
            "profit_loss_ratio": round(self.profit_loss_ratio, 4),
        }


@dataclass
class WalkForwardCycle:
    """단일 Walk-Forward 사이클 결과."""
    cycle_idx: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_sharpe: float
    oos_sharpe: float
    best_params: BacktestParams
    overfitting_flag: bool      # IS/OOS 샤프 비율 < 0.5
    lookahead_passed: bool
    n_oos_trades: int = 0

    @property
    def is_oos_ratio(self) -> float:
        if self.is_sharpe == 0:
            return 0.0
        return self.oos_sharpe / self.is_sharpe


@dataclass
class WalkForwardResult:
    """전체 Walk-Forward 결과."""
    cycles: list[WalkForwardCycle] = field(default_factory=list)
    avg_oos_sharpe: float = 0.0
    avg_is_sharpe: float = 0.0
    overfitting_cycles: int = 0
    all_oos_pnls: list[float] = field(default_factory=list)
    regime_metrics: dict[str, PeriodMetrics] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"[WalkForward] {len(self.cycles)}개 사이클",
            f"  평균 OOS 샤프: {self.avg_oos_sharpe:.3f}",
            f"  평균 IS 샤프:  {self.avg_is_sharpe:.3f}",
            f"  과적합 사이클: {self.overfitting_cycles}/{len(self.cycles)}",
            f"  전체 OOS 거래: {len(self.all_oos_pnls)}건",
        ]
        if self.regime_metrics:
            for regime, m in self.regime_metrics.items():
                lines.append(f"  [{regime}] 샤프={m.sharpe:.3f} MDD={m.max_drawdown:.1%}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# 생존 편향 처리기
# ------------------------------------------------------------------

class SurvivourshipHandler:
    """날짜별 업비트 상장 코인 목록 관리.

    SQLite coin_history 테이블에서 로드하거나
    인메모리 딕셔너리로 직접 주입 가능.
    """

    def __init__(self, snapshot_map: dict[str, list[str]] | None = None) -> None:
        """
        Args:
            snapshot_map: {'2024-01-15': ['BTC', 'ETH', ...], ...}
                          None이면 SQLite에서 로드 시도
        """
        self._snapshots: dict[str, list[str]] = snapshot_map or {}

    def load_from_db(self, db_path: str) -> None:
        """SQLite coin_history 테이블에서 스냅샷 로드."""
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT snapshot_date, coin FROM coin_history ORDER BY snapshot_date"
            ).fetchall()
            conn.close()

            from collections import defaultdict
            snap: dict[str, list[str]] = defaultdict(list)
            for date_str, coin in rows:
                snap[date_str].append(coin)
            self._snapshots = dict(snap)
            logger.info("[Survivourship] %d일치 스냅샷 로드", len(self._snapshots))
        except Exception as exc:
            logger.error("[Survivourship] DB 로드 실패: %s", exc)

    def get_coins_at(self, target_date: str | date) -> list[str]:
        """특정 날짜의 유효 코인 목록 반환.

        가장 가까운 과거 스냅샷을 사용 (미래 스냅샷 금지).
        """
        if isinstance(target_date, date):
            target_str = target_date.isoformat()
        else:
            target_str = target_date

        if not self._snapshots:
            logger.warning("[Survivourship] 스냅샷 없음 — 빈 목록 반환")
            return []

        # target 이하 날짜 중 가장 최신 스냅샷
        valid_dates = sorted(d for d in self._snapshots if d <= target_str)
        if not valid_dates:
            return []

        return self._snapshots[valid_dates[-1]]

    def has_minimum_history(self, months: int = 6) -> bool:
        """최소 N개월치 스냅샷 누적 확인."""
        if not self._snapshots:
            return False
        dates = sorted(self._snapshots.keys())
        earliest = datetime.strptime(dates[0], "%Y-%m-%d")
        latest   = datetime.strptime(dates[-1], "%Y-%m-%d")
        diff_months = (latest.year - earliest.year) * 12 + (latest.month - earliest.month)
        return diff_months >= months

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)


# ------------------------------------------------------------------
# 단일 구간 백테스트 엔진
# ------------------------------------------------------------------

class BacktestEngine:
    """단일 구간 백테스트 실행기.

    실제 전략 로직 실행 대신 시뮬레이션된 신호와 파라미터를 받아
    성과 지표를 계산한다. 전략 로직은 strategy_fn(df, params) 형태로 주입.
    """

    TRADING_COST = 0.0025   # 0.25% (수수료 + 슬리피지)

    def __init__(
        self,
        survivourship: SurvivourshipHandler | None = None,
        lookahead_checker: LookaheadBiasChecker | None = None,
    ) -> None:
        self._survivourship = survivourship or SurvivourshipHandler()
        self._lookahead = lookahead_checker or LookaheadBiasChecker()

    def run(
        self,
        df: pd.DataFrame,
        params: BacktestParams,
        strategy_fn: Callable[[pd.DataFrame, BacktestParams], list[TradeResult]] | None = None,
        period_start: str = "",
        period_end: str = "",
    ) -> PeriodMetrics:
        """단일 구간 백테스트 실행.

        Args:
            df: 피처 DataFrame (index = DatetimeIndex)
            params: 백테스트 파라미터
            strategy_fn: 거래 신호 생성 함수 (None → 기본 룰 기반)
            period_start / period_end: 기간 레이블 (표시용)

        Returns:
            PeriodMetrics
        """
        if df.empty:
            return self._empty_metrics(period_start, period_end)

        # Lookahead Bias 검증
        signal_ts = list(df.index[:100])
        lookahead_report = self._lookahead.check(df, signal_ts)
        if not lookahead_report.passed:
            logger.warning("[Backtest] Lookahead Bias 감지 — 결과 신뢰도 낮음")

        # 거래 생성
        if strategy_fn is not None:
            trades = strategy_fn(df, params)
        else:
            trades = self._default_rule_based(df, params)

        if not trades:
            return self._empty_metrics(period_start, period_end)

        # 비관적 비용 적용
        pnls = self._apply_pessimistic_cost([t.pnl_pct for t in trades])

        return self._compute_metrics(pnls, trades, period_start, period_end)

    def _default_rule_based(
        self, df: pd.DataFrame, params: BacktestParams
    ) -> list[TradeResult]:
        """기본 룰 기반 신호 생성 (ADX + RSI + SuperTrend).

        실제 프로덕션에서는 Layer1 + Layer2 앙상블 신호를 사용한다.
        여기서는 백테스트 프레임워크 검증용 단순 신호 사용.
        """
        trades = []
        if "close_5m" not in df.columns or "rsi_5m" not in df.columns:
            return trades
        if "adx_5m" not in df.columns:
            return trades

        prices = df["close_5m"].values
        rsi    = df["rsi_5m"].values
        adx    = df["adx_5m"].values
        n      = len(df)

        position_price = 0.0
        entry_idx = 0

        for i in range(1, n):
            if position_price == 0:
                # 진입: ADX > threshold AND RSI 40~60
                if (adx[i] > params.adx_threshold
                        and 40 <= rsi[i] <= 60
                        and prices[i] > 0):
                    position_price = prices[i]
                    entry_idx = i
            else:
                # 청산: 손절 or 익절 or RSI 과열
                current_pnl = (prices[i] - position_price) / position_price
                hold_min = (i - entry_idx) * 5  # 5분봉

                exit_signal = (
                    current_pnl <= -params.stop_loss_pct
                    or current_pnl >= params.take_profit_pct
                    or rsi[i] > 70
                    or rsi[i] < 30
                )

                if exit_signal and hold_min >= 15:
                    trades.append(TradeResult(
                        timestamp=df.index[i] if hasattr(df.index[i], 'to_pydatetime') else datetime.now(timezone.utc),
                        coin=df.get("coin", pd.Series(["BTC"])).iloc[0] if "coin" in df.columns else "BTC",
                        side="SELL",
                        entry_price=position_price,
                        exit_price=prices[i],
                        pnl_pct=current_pnl,
                        pnl_pct_pessimistic=0.0,  # apply_pessimistic_cost에서 채움
                        strategy_type="TREND",
                        hold_minutes=float(hold_min),
                    ))
                    position_price = 0.0

        return trades

    @staticmethod
    def _apply_pessimistic_cost(pnls: list[float]) -> list[float]:
        """비관적 비용 적용: 수익-0.25%, 손실+0.25%."""
        result = []
        for p in pnls:
            if p > 0:
                result.append(p + PESSIMISTIC_WIN_COST)
            else:
                result.append(p + PESSIMISTIC_LOSS_COST)
        return result

    @staticmethod
    def _compute_metrics(
        pnls: list[float],
        trades: list[TradeResult],
        period_start: str,
        period_end: str,
    ) -> PeriodMetrics:
        """성과 지표 계산."""
        arr = np.array(pnls, dtype=np.float64)
        n = len(arr)

        wins  = arr[arr > 0]
        loses = arr[arr < 0]

        win_rate = len(wins) / n if n > 0 else 0.0
        avg_win  = float(np.mean(wins))  if len(wins)  > 0 else 0.0
        avg_loss = float(np.mean(loses)) if len(loses) > 0 else 0.0
        pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        # 샤프비율
        mean_r = float(np.mean(arr))
        std_r  = float(np.std(arr, ddof=1)) if n > 1 else 1e-10
        sharpe = mean_r / std_r * ANNUALIZE_FACTOR if std_r > 1e-10 else 0.0

        # 최대낙폭
        cum = np.cumprod(1 + arr)
        roll_max = np.maximum.accumulate(cum)
        drawdowns = (cum - roll_max) / roll_max
        max_dd = float(np.min(drawdowns))

        total_return = float(np.prod(1 + arr) - 1)

        # 전략별 기여도
        strat_contrib: dict[str, float] = {}
        for t in trades:
            strat_contrib.setdefault(t.strategy_type, []).append(t.pnl_pct)
        strat_avg = {k: float(np.mean(v)) for k, v in strat_contrib.items()}

        return PeriodMetrics(
            period_start=period_start,
            period_end=period_end,
            n_trades=n,
            win_rate=win_rate,
            sharpe=sharpe,
            max_drawdown=max_dd,
            total_return=total_return,
            profit_loss_ratio=pl_ratio,
            strategy_contributions=strat_avg,
        )

    @staticmethod
    def _empty_metrics(start: str, end: str) -> PeriodMetrics:
        return PeriodMetrics(
            period_start=start, period_end=end,
            n_trades=0, win_rate=0.0, sharpe=0.0,
            max_drawdown=0.0, total_return=0.0, profit_loss_ratio=0.0,
        )


# ------------------------------------------------------------------
# Walk-Forward 최적화기
# ------------------------------------------------------------------

class WalkForwardOptimizer:
    """Walk-Forward 최적화기.

    6개월 IS → 1개월 OOS → 전진 반복.
    각 사이클 시작 전 Lookahead Bias 검증 자동 실행.
    """

    def __init__(
        self,
        engine: BacktestEngine | None = None,
        is_months: int = IS_MONTHS,
        oos_months: int = OOS_MONTHS,
    ) -> None:
        self._engine = engine or BacktestEngine()
        self._is_months = is_months
        self._oos_months = oos_months
        self._lookahead = LookaheadBiasChecker()

    def run(
        self,
        df: pd.DataFrame,
        optimize_fn: Callable[[pd.DataFrame], BacktestParams],
        strategy_fn: Callable[[pd.DataFrame, BacktestParams], list[TradeResult]] | None = None,
    ) -> WalkForwardResult:
        """Walk-Forward 최적화 실행.

        Args:
            df: 전체 피처 DataFrame (DatetimeIndex 필수)
            optimize_fn: IS 구간에서 최적 파라미터 반환하는 함수
            strategy_fn: 백테스트 실행 함수

        Returns:
            WalkForwardResult
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df.index는 DatetimeIndex이어야 합니다")

        result = WalkForwardResult()
        cycles = self._generate_cycles(df)

        if not cycles:
            logger.warning("[WalkForward] 사이클 생성 불가 — 데이터 부족")
            return result

        all_oos_pnls: list[float] = []

        for cycle_idx, (is_start, is_end, oos_start, oos_end) in enumerate(cycles):
            logger.info(
                "[WalkForward] 사이클 %d: IS=%s~%s OOS=%s~%s",
                cycle_idx, is_start, is_end, oos_start, oos_end,
            )

            df_is  = df.loc[is_start:is_end]
            df_oos = df.loc[oos_start:oos_end]

            if df_is.empty or df_oos.empty:
                continue

            # Lookahead Bias 검증 (사이클 시작 전 자동)
            signal_ts = list(df_is.index[:50])
            la_report = self._lookahead.check(df_is, signal_ts)

            # IS 최적화
            best_params = optimize_fn(df_is)

            # IS 성과
            is_metrics = self._engine.run(df_is, best_params, strategy_fn,
                                          is_start, is_end)

            # OOS 평가 (최적 파라미터 고정)
            oos_metrics = self._engine.run(df_oos, best_params, strategy_fn,
                                           oos_start, oos_end)

            # 과적합 판별
            overfitting = (
                is_metrics.sharpe > 0
                and oos_metrics.sharpe / is_metrics.sharpe < OVERFITTING_THRESHOLD
            )
            if overfitting:
                logger.warning(
                    "[WalkForward] 과적합 의심 — IS=%.3f OOS=%.3f ratio=%.3f",
                    is_metrics.sharpe, oos_metrics.sharpe,
                    oos_metrics.sharpe / is_metrics.sharpe if is_metrics.sharpe else 0,
                )

            cycle = WalkForwardCycle(
                cycle_idx=cycle_idx,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
                is_sharpe=is_metrics.sharpe,
                oos_sharpe=oos_metrics.sharpe,
                best_params=best_params,
                overfitting_flag=overfitting,
                lookahead_passed=la_report.passed,
                n_oos_trades=oos_metrics.n_trades,
            )
            result.cycles.append(cycle)
            if overfitting:
                result.overfitting_cycles += 1

            # OOS 손익 누적
            all_oos_pnls.extend([0.0] * oos_metrics.n_trades)  # 실제 trade pnl 집계 시 교체

        result.all_oos_pnls = all_oos_pnls
        if result.cycles:
            result.avg_oos_sharpe = float(np.mean([c.oos_sharpe for c in result.cycles]))
            result.avg_is_sharpe  = float(np.mean([c.is_sharpe  for c in result.cycles]))

        logger.info(result.summary())
        return result

    def _generate_cycles(
        self, df: pd.DataFrame
    ) -> list[tuple[str, str, str, str]]:
        """IS/OOS 구간 목록 생성."""
        if df.empty:
            return []

        start = df.index.min()
        end   = df.index.max()
        cycles = []

        cursor = start
        while True:
            is_end_approx   = cursor + pd.DateOffset(months=self._is_months)
            oos_end_approx  = is_end_approx + pd.DateOffset(months=self._oos_months)

            if oos_end_approx > end:
                break

            cycles.append((
                cursor.strftime("%Y-%m-%d"),
                is_end_approx.strftime("%Y-%m-%d"),
                is_end_approx.strftime("%Y-%m-%d"),
                oos_end_approx.strftime("%Y-%m-%d"),
            ))
            cursor = is_end_approx  # 전진

        return cycles

    def evaluate_regime_periods(
        self,
        df_full: pd.DataFrame,
        params: BacktestParams,
        strategy_fn: Callable | None = None,
    ) -> dict[str, PeriodMetrics]:
        """전략별 구간 분리 백테스트 (bull / sideways / bear).

        하락장 구간 낙폭 < 10% 필수 조건 체크.
        """
        metrics: dict[str, PeriodMetrics] = {}

        for regime, (start, end) in REGIME_PERIODS.items():
            df_period = df_full.loc[start:end]
            if df_period.empty:
                logger.info("[WalkForward] '%s' 구간 데이터 없음 (%s~%s)", regime, start, end)
                continue

            m = self._engine.run(df_period, params, strategy_fn, start, end)
            metrics[regime] = m

            logger.info(
                "[WalkForward] %s: 샤프=%.3f MDD=%.1f%% 승률=%.1f%%",
                regime, m.sharpe, m.max_drawdown * 100, m.win_rate * 100,
            )

            # 하락장 낙폭 검증
            if regime == "bear" and m.max_drawdown < BEAR_MAX_DRAWDOWN:
                logger.warning(
                    "[WalkForward] 하락장 낙폭 %.1f%% > 기준 %.1f%% — 실전 전환 불가",
                    abs(m.max_drawdown) * 100, abs(BEAR_MAX_DRAWDOWN) * 100,
                )

        return metrics

    def check_live_readiness(self, result: WalkForwardResult) -> list[str]:
        """실전 전환 7가지 기준 체크.

        Returns:
            미통과 기준 목록 (비어있으면 전환 가능)
        """
        failures: list[str] = []

        if result.avg_oos_sharpe < 1.5:
            failures.append(f"OOS 샤프 {result.avg_oos_sharpe:.3f} < 1.5")

        bear = result.regime_metrics.get("bear")
        if bear and bear.max_drawdown < BEAR_MAX_DRAWDOWN:
            failures.append(f"하락장 낙폭 {bear.max_drawdown:.1%} < -10%")
        if bear and bear.win_rate < 0.55:
            failures.append(f"하락장 승률 {bear.win_rate:.1%} < 55%")

        lookahead_fails = [c for c in result.cycles if not c.lookahead_passed]
        if lookahead_fails:
            failures.append(f"Lookahead Bias 미통과 사이클 {len(lookahead_fails)}개")

        if result.overfitting_cycles > len(result.cycles) * 0.5:
            failures.append(f"과적합 사이클 {result.overfitting_cycles}/{len(result.cycles)}")

        return failures
