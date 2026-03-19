"""
risk/trailing_stop.py — ATR 기반 트레일링 스탑 관리

[설계]
  - 포지션별 최고가(peak) 추적 → ATR×배수 이하 이탈 시 청산 신호
  - 부분 익절: 목표가 도달 시 50% 매도 (선택 활성화)
  - 레짐별 멀티플라이어:
      TREND_STRONG  → 3.0 × ATR (여유)
      TREND_NORMAL  → 2.5 × ATR
      GRID / DCA    → 2.0 × ATR (타이트)

사용법:
    ts = TrailingStopManager()
    ts.init(coin="KRW-BTC", entry_price=50_000_000, atr=750_000, regime="TREND_STRONG")
    stop_price = ts.get_stop(coin)            # 현재 스탑 가격
    triggered = ts.update(coin, current_price=51_000_000, atr=750_000)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

REGIME_MULTIPLIER: dict[str, float] = {
    "TREND_STRONG": 3.0,
    "TREND_NORMAL": 2.5,
    "GRID":         2.0,
    "DCA":          2.0,
    "HOLD":         2.5,
}
DEFAULT_MULTIPLIER = 2.5

# 부분 익절 목표 (진입가 대비 %, 양수)
PARTIAL_EXIT_TARGET_PCT = 0.05   # 5% 상승 시
PARTIAL_EXIT_RATIO       = 0.50  # 50% 매도


# ------------------------------------------------------------------
# 포지션별 스탑 상태
# ------------------------------------------------------------------

@dataclass
class _StopState:
    entry_price:      float
    peak_price:       float
    stop_price:       float
    atr_multiplier:   float
    partial_done:     bool = False
    created_at:       datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ------------------------------------------------------------------
# TrailingStopManager
# ------------------------------------------------------------------

class TrailingStopManager:
    """모든 코인 포지션의 트레일링 스탑을 통합 관리.

    TradingEngine.position_loop()에서 1분마다 update()를 호출.
    """

    def __init__(self) -> None:
        self._states: dict[str, _StopState] = {}

    # ------------------------------------------------------------------
    # 포지션 등록 / 해제
    # ------------------------------------------------------------------

    def init(
        self,
        coin: str,
        entry_price: float,
        atr: float,
        regime: str = "TREND_NORMAL",
    ) -> float:
        """새 포지션의 트레일링 스탑 초기화.

        Returns:
            초기 스탑 가격.
        """
        mult = REGIME_MULTIPLIER.get(regime, DEFAULT_MULTIPLIER)
        stop = entry_price - atr * mult
        state = _StopState(
            entry_price=entry_price,
            peak_price=entry_price,
            stop_price=stop,
            atr_multiplier=mult,
        )
        self._states[coin] = state
        logger.info(
            "[TrailingStop] 초기화 %s entry=%.0f stop=%.0f (ATR×%.1f)",
            coin, entry_price, stop, mult,
        )
        return stop

    def remove(self, coin: str) -> None:
        """포지션 청산 시 스탑 상태 삭제."""
        self._states.pop(coin, None)

    def clear(self) -> None:
        """전체 포지션 초기화."""
        self._states.clear()

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------

    def get_stop(self, coin: str) -> float | None:
        """현재 스탑 가격 반환. 미등록 → None."""
        s = self._states.get(coin)
        return s.stop_price if s else None

    def get_partial_ratio(self, coin: str, current_price: float) -> float | None:
        """부분 익절 비율 반환.

        부분 익절 조건 충족 시 PARTIAL_EXIT_RATIO(0.5), 아니면 None.
        """
        s = self._states.get(coin)
        if s is None or s.partial_done:
            return None
        target = s.entry_price * (1 + PARTIAL_EXIT_TARGET_PCT)
        if current_price >= target:
            return PARTIAL_EXIT_RATIO
        return None

    def has_position(self, coin: str) -> bool:
        return coin in self._states

    # ------------------------------------------------------------------
    # 가격 업데이트 → 스탑 발동 여부
    # ------------------------------------------------------------------

    def update(self, coin: str, current_price: float, atr: float) -> bool:
        """최신 가격으로 스탑 업데이트.

        Returns:
            True — 스탑 발동 (청산 필요)
            False — 정상 유지
        """
        s = self._states.get(coin)
        if s is None:
            return False

        now = datetime.now(timezone.utc)

        # Peak 갱신
        if current_price > s.peak_price:
            s.peak_price = current_price

            # 스탑 상향 조정
            new_stop = s.peak_price - atr * s.atr_multiplier
            if new_stop > s.stop_price:
                logger.debug(
                    "[TrailingStop] %s 스탑 상향 %.0f → %.0f (peak=%.0f)",
                    coin, s.stop_price, new_stop, s.peak_price,
                )
                s.stop_price = new_stop

        s.last_updated_at = now

        # 스탑 발동 체크
        if current_price <= s.stop_price:
            logger.info(
                "[TrailingStop] 발동 %s price=%.0f <= stop=%.0f",
                coin, current_price, s.stop_price,
            )
            return True

        return False

    def mark_partial_done(self, coin: str) -> None:
        """부분 익절 완료 표시."""
        s = self._states.get(coin)
        if s:
            s.partial_done = True

    # ------------------------------------------------------------------
    # 요약
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, dict]:
        """전체 포지션 스탑 요약."""
        return {
            coin: {
                "entry":   s.entry_price,
                "peak":    s.peak_price,
                "stop":    s.stop_price,
                "mult":    s.atr_multiplier,
                "partial": s.partial_done,
            }
            for coin, s in self._states.items()
        }
