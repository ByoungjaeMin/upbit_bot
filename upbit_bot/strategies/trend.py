"""
strategies/trend.py — 추세 추종 전략 (TREND_STRONG / TREND_NORMAL)

[진입 조건]
  공통:
    - Layer2 signal_confirmed=True
    - Layer1 tradeable=True, regime_strategy in ('TREND_STRONG', 'TREND_NORMAL')
    - CircuitBreaker level == 0
    - SuperTrend 방향 +1 (상승)
    - RSI 45~75 범위 (과매수/과매도 배제)

  TREND_STRONG 추가:
    - ADX ≥ 30
    - 멀티타임프레임 3/3 합의 (5m·1h·1d 모두 상승)

  TREND_NORMAL 추가:
    - ADX 20~30
    - 멀티타임프레임 2/3 이상 합의

[청산 조건]
  - TrailingStop 발동 (ATR×3.0 / 2.5 이하)
  - SuperTrend 반전 (-1)
  - RSI < 40 (모멘텀 소실)
  - Layer1 reversal_detected=True

[포지션 관리]
  - 부분 익절: 진입가 +5% 도달 시 50% 매도
  - 최대 보유 시간: TREND_STRONG=24시간, TREND_NORMAL=12시간
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from schema import FilterResult, EnsemblePrediction, MarketState

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

RSI_ENTRY_MIN   = 45.0
RSI_ENTRY_MAX   = 75.0
RSI_EXIT_MIN    = 40.0
ADX_STRONG_MIN  = 30.0
ADX_NORMAL_MIN  = 20.0
MAX_HOLD_STRONG = timedelta(hours=24)
MAX_HOLD_NORMAL = timedelta(hours=12)
PARTIAL_TARGET_PCT = 0.05   # 5% 상승 시 부분 익절


# ------------------------------------------------------------------
# 진입 신호
# ------------------------------------------------------------------

@dataclass
class TrendSignal:
    """추세 전략 진입/청산 신호."""
    coin: str
    action: str           # 'ENTER' | 'EXIT' | 'PARTIAL_EXIT' | 'HOLD'
    strategy_type: str    # 'TREND_STRONG' | 'TREND_NORMAL'
    reason: str           # 신호 사유 (로깅용)
    confidence: float     # 0~1 (앙상블 weighted_avg)


class TrendStrategy:
    """추세 추종 전략 평가기.

    Layer1 FilterResult + Layer2 EnsemblePrediction + MarketState를 조합해
    진입/청산/홀딩 신호를 생성.

    사용법:
        strategy = TrendStrategy()
        signal = strategy.evaluate_entry(filter_result, ensemble, market_state)
        exit_sig = strategy.evaluate_exit(coin, market_state, entry_price, entry_time)
    """

    # ------------------------------------------------------------------
    # 진입 평가
    # ------------------------------------------------------------------

    def evaluate_entry(
        self,
        fr: "FilterResult",
        ep: "EnsemblePrediction",
        ms: "MarketState",
    ) -> TrendSignal | None:
        """진입 조건 평가. 신호 없으면 None 반환."""
        if not fr.tradeable:
            return None
        if fr.regime_strategy not in ("TREND_STRONG", "TREND_NORMAL"):
            return None
        if not ep.signal_confirmed:
            return None

        # RSI 범위 체크
        rsi = ms.rsi_5m
        if not (RSI_ENTRY_MIN <= rsi <= RSI_ENTRY_MAX):
            logger.debug("[Trend] %s RSI=%.1f 범위 이탈 — 진입 스킵", fr.coin, rsi)
            return None

        # SuperTrend 방향
        if ms.supertrend_signal != 1:
            logger.debug("[Trend] %s SuperTrend 하락 — 진입 스킵", fr.coin)
            return None

        strategy_type = fr.regime_strategy
        adx = ms.adx_5m

        if strategy_type == "TREND_STRONG":
            if adx < ADX_STRONG_MIN:
                logger.debug("[Trend] %s ADX=%.1f < %s — 강한추세 조건 미충족", fr.coin, adx, ADX_STRONG_MIN)
                return None
            reason = f"TREND_STRONG 진입 ADX={adx:.1f} RSI={rsi:.1f} ensemble={ep.weighted_avg:.3f}"
        else:  # TREND_NORMAL
            if adx < ADX_NORMAL_MIN:
                logger.debug("[Trend] %s ADX=%.1f < %s — 일반추세 조건 미충족", fr.coin, adx, ADX_NORMAL_MIN)
                return None
            reason = f"TREND_NORMAL 진입 ADX={adx:.1f} RSI={rsi:.1f} ensemble={ep.weighted_avg:.3f}"

        logger.info("[Trend] %s %s", fr.coin, reason)
        return TrendSignal(
            coin=fr.coin,
            action="ENTER",
            strategy_type=strategy_type,
            reason=reason,
            confidence=ep.weighted_avg,
        )

    # ------------------------------------------------------------------
    # 청산 평가
    # ------------------------------------------------------------------

    def evaluate_exit(
        self,
        coin: str,
        ms: "MarketState",
        entry_price: float,
        entry_time: datetime,
        strategy_type: str = "TREND_NORMAL",
        trailing_stop_triggered: bool = False,
        reversal_detected: bool = False,
    ) -> TrendSignal | None:
        """청산 조건 평가. 청산 신호 없으면 None 반환."""
        now = datetime.now(timezone.utc)

        # 1. 트레일링 스탑 발동
        if trailing_stop_triggered:
            return TrendSignal(
                coin=coin, action="EXIT",
                strategy_type=strategy_type,
                reason="트레일링 스탑 발동",
                confidence=1.0,
            )

        # 2. SuperTrend 반전
        if ms.supertrend_signal == -1:
            return TrendSignal(
                coin=coin, action="EXIT",
                strategy_type=strategy_type,
                reason=f"SuperTrend 하락 반전",
                confidence=0.9,
            )

        # 3. RSI 모멘텀 소실
        if ms.rsi_5m < RSI_EXIT_MIN:
            return TrendSignal(
                coin=coin, action="EXIT",
                strategy_type=strategy_type,
                reason=f"RSI={ms.rsi_5m:.1f} < {RSI_EXIT_MIN} 모멘텀 소실",
                confidence=0.8,
            )

        # 4. Layer1 반전 감지
        if reversal_detected:
            return TrendSignal(
                coin=coin, action="EXIT",
                strategy_type=strategy_type,
                reason="Layer1 반전 감지",
                confidence=0.75,
            )

        # 5. 최대 보유 시간 초과
        max_hold = MAX_HOLD_STRONG if strategy_type == "TREND_STRONG" else MAX_HOLD_NORMAL
        if now - entry_time > max_hold:
            return TrendSignal(
                coin=coin, action="EXIT",
                strategy_type=strategy_type,
                reason=f"최대 보유 시간 초과 ({max_hold})",
                confidence=0.7,
            )

        # 6. 부분 익절 체크
        if entry_price > 0:
            pnl_pct = (ms.close_5m - entry_price) / entry_price
            if pnl_pct >= PARTIAL_TARGET_PCT:
                return TrendSignal(
                    coin=coin, action="PARTIAL_EXIT",
                    strategy_type=strategy_type,
                    reason=f"부분 익절 목표 도달 (+{pnl_pct:.1%})",
                    confidence=0.85,
                )

        return None
