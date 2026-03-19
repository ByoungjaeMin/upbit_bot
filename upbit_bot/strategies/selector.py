"""
strategies/selector.py — StrategySelector

HMM 레짐(Phase C) 또는 ADX(Phase A/B) 기반 3전략 자동 전환.
StrategyDecayMonitor 연동으로 성과 기반 동적 자본 배분 지원 (Phase C).

레짐 → 전략 매핑:
  레짐 0 (ADX>30):  TREND_STRONG  — 추세추종 + 피라미딩 (자본 60%)
  레짐 1 (ADX20~30): TREND_NORMAL — 추세추종 보수적 (자본 60%)
  레짐 2 (ADX<20):  GRID          — 횡보 그리드 (자본 20%)
  레짐 3 (F&G<30):  DCA           — 적응형 DCA (자본 15%)
  레짐 3 (F&G>=30): HOLD          — USDT 유지 (자본 5%)

동적 자본 배분 (Phase C, decay_monitor 연동):
  최종 배분 = 기본 배분 × 0.5 + 동적 배분 × 0.5
  모든 전략 DORMANT → HOLD 100% 폴백
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from schema import FilterResult, StrategyDecision

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 기본 자본 배분 (성과 데이터 없을 때 적용)
# ------------------------------------------------------------------

BASE_ALLOCATION: dict[str, float] = {
    "TREND_STRONG": 0.60,
    "TREND_NORMAL": 0.60,
    "GRID":         0.20,
    "DCA":          0.15,
    "HOLD":         0.05,
}

# ADX 임계값 (Phase A/B)
ADX_STRONG = 30.0
ADX_NORMAL = 20.0


class StrategySelector:
    """HMM 레짐 / ADX 기반 전략 선택기.

    사용법:
        selector = StrategySelector()
        decision = selector.select_strategy(
            adx=28.0,
            fear_greed=45.0,
            coin="KRW-BTC",
            filter_result=fr,
        )
    """

    def __init__(
        self,
        decay_monitor: Any = None,   # StrategyDecayMonitor (Phase C)
        phase_c_enabled: bool = False,
    ) -> None:
        self._decay_monitor = decay_monitor
        self._phase_c = phase_c_enabled

    def select_strategy(
        self,
        adx: float,
        fear_greed: float,
        coin: str = "",
        hmm_regime: int = -1,
        filter_result: FilterResult | None = None,
    ) -> StrategyDecision:
        """레짐 → 전략 결정 + 자본 배분 계산.

        Args:
            adx: 5분봉 ADX 값
            fear_greed: Fear&Greed 지수 (0~100)
            coin: 대상 코인
            hmm_regime: HMM 레짐 (Phase C). -1이면 ADX 기반 폴백.
            filter_result: Layer 1 출력 (signal_multiplier 반영)
        """
        strategy_type = self._resolve_strategy(adx, fear_greed, hmm_regime)
        base_alloc = BASE_ALLOCATION[strategy_type]

        # signal_multiplier 반영
        mult = filter_result.signal_multiplier if filter_result else 1.0
        allocation = base_alloc * mult

        # Phase C: 동적 자본 배분 연동
        if self._phase_c and self._decay_monitor is not None:
            allocation = self._apply_dynamic_weights(strategy_type, base_alloc, mult)

        allocation = round(max(0.0, min(1.0, allocation)), 4)

        grid_params = self._build_grid_params(filter_result) if strategy_type in ("GRID",) else None
        dca_params = self._build_dca_params() if strategy_type == "DCA" else None

        decision = StrategyDecision(
            coin=coin,
            timestamp=datetime.now(timezone.utc),
            strategy_type=strategy_type,
            capital_allocation=allocation,
            grid_params=grid_params,
            dca_params=dca_params,
            dynamic_weight=mult,
        )
        logger.debug(
            "[StrategySelector] %s → %s alloc=%.2f%% (ADX=%.1f, F&G=%.0f)",
            coin, strategy_type, allocation * 100, adx, fear_greed,
        )
        return decision

    # ------------------------------------------------------------------
    # 레짐 → 전략 타입
    # ------------------------------------------------------------------

    def _resolve_strategy(
        self, adx: float, fear_greed: float, hmm_regime: int
    ) -> str:
        """HMM(Phase C) 또는 ADX(Phase A/B) 기반 전략 결정."""
        if self._phase_c and hmm_regime >= 0:
            return self._hmm_to_strategy(hmm_regime, adx, fear_greed)
        return self._adx_to_strategy(adx, fear_greed)

    @staticmethod
    def _adx_to_strategy(adx: float, fear_greed: float) -> str:
        """Phase A/B: ADX 기반 전략 결정."""
        if adx >= ADX_STRONG:
            return "TREND_STRONG"
        if adx >= ADX_NORMAL:
            return "TREND_NORMAL"
        # 횡보 구간
        if fear_greed < 30:
            return "DCA"
        return "GRID"

    @staticmethod
    def _hmm_to_strategy(hmm_regime: int, adx: float, fear_greed: float) -> str:
        """Phase C: HMM 레짐 기반 전략 결정."""
        if hmm_regime == 0:
            return "TREND_STRONG" if adx >= ADX_STRONG else "TREND_NORMAL"
        if hmm_regime == 1:
            return "TREND_NORMAL"
        if hmm_regime == 2:
            return "GRID"
        # hmm_regime == 3
        return "DCA" if fear_greed < 30 else "HOLD"

    # ------------------------------------------------------------------
    # 동적 자본 배분 (Phase C)
    # ------------------------------------------------------------------

    def _apply_dynamic_weights(
        self, strategy_type: str, base_alloc: float, mult: float
    ) -> float:
        """decay_monitor.get_weights() 연동 동적 배분.

        최종 = 기본 × 0.5 + 동적 × 0.5
        모든 전략 DORMANT → HOLD 100% 폴백
        """
        try:
            weights = self._decay_monitor.get_weights()
        except Exception as exc:
            logger.warning("[StrategySelector] decay_monitor 오류 — 기본 배분 사용: %s", exc)
            return base_alloc * mult

        if not weights or all(v == 0.0 for v in weights.values()):
            # 모든 전략 DORMANT → HOLD 100%
            logger.warning("[StrategySelector] 모든 전략 DORMANT — HOLD 100%")
            return BASE_ALLOCATION["HOLD"]

        dynamic_alloc = weights.get(strategy_type, base_alloc)
        blended = base_alloc * 0.5 + dynamic_alloc * 0.5
        return blended * mult

    # ------------------------------------------------------------------
    # 전략별 파라미터 생성
    # ------------------------------------------------------------------

    @staticmethod
    def _build_grid_params(fr: FilterResult | None) -> dict[str, Any]:
        return {
            "levels": 10,
            "atr_multiplier": 3.0,   # range = price ± ATR×3
            "atr_value": fr.atr_value if fr else 0.0,
        }

    @staticmethod
    def _build_dca_params() -> dict[str, Any]:
        return {
            "step_pct": -0.03,       # -3% 하락마다 Safety Order
            "volume_scale": 1.5,     # 매수금액 1.5배씩 증가
            "max_safety": 5,
            "take_profit_pct": 0.03, # +3% 익절
        }
