"""
risk/kelly.py — Kelly Fractional + VaR 포지션 사이징

[설계]
  1. 단일 자산 Kelly f* = (p*b - q) / b  (p=승률, b=손익비, q=1-p)
  2. HMM 레짐 신뢰도 × Kelly 조정 (Phase C에서 hmm_confidence 활용)
  3. 역사적 VaR 95% 오버레이 — 포지션 크기 추가 제한
  4. 최대 단일 포지션 30% (MAX_SINGLE_PCT) 하드 캡
  5. 연속 손실 패널티 (consecutive_losses × 10% 추가 감소)
  6. 최소 투자금 50,000 KRW (MIN_POSITION_KRW)

[출력]
  RiskBudget dataclass (schema.py)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Sequence

from schema import RiskBudget

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

KELLY_FRACTION       = 0.25       # 풀 Kelly의 1/4 (보수적)
MAX_SINGLE_PCT       = 0.30       # 단일 포지션 최대 자본 30%
MIN_POSITION_KRW     = 50_000     # 최소 투자금 KRW
VAR_CONFIDENCE       = 0.95       # 역사적 VaR 신뢰 수준
CONSECUTIVE_PENALTY  = 0.10       # 연속 손실 1회당 추가 10% 감소
MAX_CONSECUTIVE_PEN  = 0.50       # 최대 연속 손실 패널티 50%

# ATR 변동성 그룹 기준
ATR_HIGH_THRESHOLD   = 0.03       # ATR/가격 > 3% → HIGH
ATR_LOW_THRESHOLD    = 0.005      # ATR/가격 < 0.5% → LOW


# ------------------------------------------------------------------
# KellySizer
# ------------------------------------------------------------------

class KellySizer:
    """Kelly Fractional + VaR 기반 포지션 사이저.

    사용법:
        sizer = KellySizer(total_capital=10_000_000)
        budget = sizer.compute(
            coin="KRW-BTC",
            win_rate=0.55,
            profit_loss_ratio=1.8,
            hmm_confidence=0.72,
            atr_price_ratio=0.015,
            recent_returns=[0.02, -0.01, 0.03, ...],
            consecutive_losses=0,
        )
    """

    def __init__(
        self,
        total_capital: float,
        kelly_fraction: float = KELLY_FRACTION,
        max_single_pct: float = MAX_SINGLE_PCT,
    ) -> None:
        self._capital       = total_capital
        self._kelly_frac    = kelly_fraction
        self._max_single    = max_single_pct

    def update_capital(self, new_capital: float) -> None:
        """자본 동적 업데이트 (매 사이클 호출)."""
        self._capital = new_capital

    def compute(
        self,
        coin: str,
        win_rate: float,
        profit_loss_ratio: float,
        hmm_confidence: float = 0.0,
        hmm_regime: int = -1,
        atr_price_ratio: float = 0.01,
        recent_returns: Sequence[float] | None = None,
        consecutive_losses: int = 0,
    ) -> RiskBudget:
        """RiskBudget 계산.

        Args:
            coin: 코인 식별자 ('KRW-BTC' 등)
            win_rate: 승률 (0~1)
            profit_loss_ratio: 손익비 b (평균이익/평균손실)
            hmm_confidence: HMM 신뢰도 (0~1, Phase A=0)
            hmm_regime: HMM 레짐 (-1=미사용)
            atr_price_ratio: ATR/현재가 (변동성 그룹 결정)
            recent_returns: 최근 N일 일일 수익률 리스트 (VaR 계산용)
            consecutive_losses: 연속 손실 횟수
        """
        ts = datetime.now(timezone.utc)

        # 1. 순수 Kelly f*
        raw_kelly = self._raw_kelly(win_rate, profit_loss_ratio)

        # 2. 분수 Kelly (보수적)
        fractional_kelly = raw_kelly * self._kelly_frac

        # 3. HMM 신뢰도 조정
        hmm_adjusted = self._hmm_adjust(fractional_kelly, hmm_confidence, hmm_regime)

        # 4. VaR 오버레이
        var_95 = self._historical_var(recent_returns or [])
        var_adjusted = self._var_overlay(hmm_adjusted, var_95)

        # 5. 연속 손실 패널티
        penalty = min(consecutive_losses * CONSECUTIVE_PENALTY, MAX_CONSECUTIVE_PEN)
        penalized = var_adjusted * (1.0 - penalty)

        # 6. 하드 캡 + 최소 포지션
        final_f = min(max(penalized, 0.0), self._max_single)
        final_size = self._capital * final_f
        if 0 < final_size < MIN_POSITION_KRW:
            final_size = 0.0  # 최소금액 미달 → 포지션 없음

        # ATR 변동성 그룹
        coin_group = _atr_group(atr_price_ratio)

        logger.debug(
            "[KellySizer] %s raw=%.4f frac=%.4f hmm=%.4f var=%.4f final=%.4f"
            " size=%.0f",
            coin, raw_kelly, fractional_kelly, hmm_adjusted, var_adjusted,
            final_f, final_size,
        )

        return RiskBudget(
            coin=coin,
            timestamp=ts,
            kelly_f=raw_kelly,
            hmm_adjusted_f=hmm_adjusted,
            var_adjusted_f=var_adjusted,
            final_position_size=final_size,
            coin_group=coin_group,
            consecutive_losses=consecutive_losses,
            var_95=var_95,
        )

    # ------------------------------------------------------------------
    # 내부 계산
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_kelly(win_rate: float, profit_loss_ratio: float) -> float:
        """단일 자산 Kelly f* = (p*b - q) / b."""
        p = max(0.0, min(1.0, win_rate))
        b = max(0.0, profit_loss_ratio)
        if b == 0:
            return 0.0
        q = 1.0 - p
        f = (p * b - q) / b
        return max(0.0, f)

    @staticmethod
    def _hmm_adjust(f: float, confidence: float, regime: int) -> float:
        """HMM 신뢰도 × f.

        Phase A/B: confidence=0 → 조정 없음 (f 그대로).
        Phase C: 레짐별 추가 조정.
          regime=0 (강한 상승) → ×1.1 보너스 (최대 MAX_SINGLE_PCT 이내)
          regime=3 (하락)     → ×0.5 추가 감소
        """
        if confidence <= 0 or regime < 0:
            return f

        base = f * (0.5 + 0.5 * confidence)  # 신뢰도 낮으면 절반으로 시작

        if regime == 0:
            return min(base * 1.1, MAX_SINGLE_PCT)
        if regime == 3:
            return base * 0.5
        return base

    @staticmethod
    def _historical_var(returns: Sequence[float], confidence: float = VAR_CONFIDENCE) -> float:
        """역사적 VaR (자본 대비 %).

        returns: 일일 수익률 리스트 (예: [-0.02, 0.01, ...])
        confidence: 신뢰 수준 (기본 0.95)
        반환: VaR 절댓값 (양수, 예: 0.03 = 3%)
        """
        if not returns:
            return 0.02  # 기본 2%

        sorted_returns = sorted(returns)
        idx = max(0, int(math.floor((1 - confidence) * len(sorted_returns))) - 1)
        var = -sorted_returns[idx]
        return max(var, 0.0)

    @staticmethod
    def _var_overlay(f: float, var_95: float) -> float:
        """VaR 95% 초과 시 포지션 축소.

        포지션 f 가 VaR의 2배를 넘으면 VaR×2 로 제한.
        """
        if var_95 <= 0:
            return f
        var_cap = var_95 * 2.0
        return min(f, var_cap)


# ------------------------------------------------------------------
# ATR 변동성 그룹
# ------------------------------------------------------------------

def _atr_group(atr_price_ratio: float) -> str:
    if atr_price_ratio > ATR_HIGH_THRESHOLD:
        return "HIGH"
    if atr_price_ratio < ATR_LOW_THRESHOLD:
        return "LOW"
    return "NORMAL"
