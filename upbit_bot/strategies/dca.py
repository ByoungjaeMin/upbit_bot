"""
strategies/dca.py — AdaptiveDCAStrategy (적응형 Dollar Cost Averaging)

Fear&Greed 연동 동적 매수 규모 조정.
Safety Order 기하급수 증가 (base × 1.5^n).
+3% 평균 매수가 도달 시 전량 익절.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# F&G 임계값
FG_EXTREME_FEAR = 15
FG_FEAR = 30

# 기본 파라미터
DEFAULT_STEP_PCT = -0.03      # -3% 하락마다 Safety Order
DEFAULT_VOLUME_SCALE = 1.5   # 매수금액 1.5배씩 증가
DEFAULT_MAX_SAFETY = 5
DEFAULT_TP_PCT = 0.03         # +3% 익절


@dataclass
class DCAOrder:
    """단일 DCA 매수 기록."""
    order_n: int               # 0=초기, 1~5=Safety
    price: float               # 체결가 (KRW)
    krw_amount: float          # 투입 금액 (KRW)
    coin_qty: float            # 매수 수량
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdaptiveDCAStrategy:
    """적응형 DCA 전략.

    사용법:
        dca = AdaptiveDCAStrategy(capital=300_000)
        # 초기 매수
        dca.add_safety_order(current_price=50_000_000)
        # -3% 하락 후
        dca.add_safety_order(current_price=48_500_000)
        # 익절 조건 확인
        if dca.check_take_profit(current_price):
            # 전량 청산
    """

    def __init__(
        self,
        capital: float,
        step_pct: float = DEFAULT_STEP_PCT,
        volume_scale: float = DEFAULT_VOLUME_SCALE,
        max_safety: int = DEFAULT_MAX_SAFETY,
        take_profit_pct: float = DEFAULT_TP_PCT,
    ) -> None:
        self._capital = capital
        self._base_amount = capital * 0.02   # 자본의 2% = 초기 매수금
        self._step_pct = step_pct
        self._volume_scale = volume_scale
        self._max_safety = max_safety
        self._take_profit_pct = take_profit_pct

        self._orders: list[DCAOrder] = []
        self._last_buy_price: float = 0.0

    # ------------------------------------------------------------------
    # Safety Order 추가
    # ------------------------------------------------------------------

    def add_safety_order(
        self,
        current_price: float,
        fear_greed: float = 50.0,
    ) -> DCAOrder | None:
        """현재가 기준 Safety Order 추가.

        조건: 이전 매수가 대비 step_pct(-3%) 이하 하락 OR 첫 매수.
        매수 금액 = base_amount × volume_scale^n × fg_multiplier

        Returns:
            생성된 DCAOrder (조건 미충족 or 상한 초과 시 None)
        """
        n = len(self._orders)

        if n >= self._max_safety + 1:  # 초기 1 + Safety 5 = 최대 6
            logger.debug("[DCA] Safety Order 상한 초과 (n=%d)", n)
            return None

        # 첫 매수가 아닐 때: 하락 조건 확인
        if self._last_buy_price > 0:
            change = (current_price - self._last_buy_price) / self._last_buy_price
            if change > self._step_pct:
                logger.debug(
                    "[DCA] 하락폭 부족 (change=%.2f%% > step=%.2f%%)",
                    change * 100, self._step_pct * 100,
                )
                return None

        # 매수 금액: n=0(초기)→base×scale^0, n=1(safety1)→base×scale^1 ...
        amount = self._base_amount * (self._volume_scale ** n)
        amount *= self.fear_greed_multiplier(fear_greed)
        amount = min(amount, self._capital * 0.5)  # 자본 50% 상한

        if current_price <= 0:
            return None

        coin_qty = amount / current_price
        order = DCAOrder(
            order_n=n,
            price=current_price,
            krw_amount=amount,
            coin_qty=coin_qty,
        )
        self._orders.append(order)
        self._last_buy_price = current_price

        logger.info(
            "[DCA] Safety Order %d: price=%.0f amount=%.0f qty=%.6f",
            n, current_price, amount, coin_qty,
        )
        return order

    # ------------------------------------------------------------------
    # 평균 매수가
    # ------------------------------------------------------------------

    def avg_entry_price(self) -> float:
        """가중 평균 매수가 (총 투입금액 / 총 수량)."""
        total_qty = sum(o.coin_qty for o in self._orders)
        total_krw = sum(o.krw_amount for o in self._orders)
        if total_qty <= 0:
            return 0.0
        return total_krw / total_qty

    # ------------------------------------------------------------------
    # 익절 조건
    # ------------------------------------------------------------------

    def check_take_profit(self, current_price: float) -> bool:
        """평균 매수가 대비 +take_profit_pct(기본+3%) → 전량 익절 신호."""
        avg = self.avg_entry_price()
        if avg <= 0:
            return False
        return current_price >= avg * (1 + self._take_profit_pct)

    # ------------------------------------------------------------------
    # Fear&Greed 배율
    # ------------------------------------------------------------------

    @staticmethod
    def fear_greed_multiplier(fear_greed: float) -> float:
        """F&G 지수에 따른 매수 규모 배율.

        극단적 공포(< 15): ×1.5 — 저점 공격적 매수
        공포(< 30):        ×1.0 — 표준
        그 외:             ×0.5 — 과열 구간 매수 자제
        """
        if fear_greed < FG_EXTREME_FEAR:
            return 1.5
        if fear_greed < FG_FEAR:
            return 1.0
        return 0.5

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------

    @property
    def order_count(self) -> int:
        return len(self._orders)

    @property
    def total_invested_krw(self) -> float:
        return sum(o.krw_amount for o in self._orders)

    @property
    def total_coin_qty(self) -> float:
        return sum(o.coin_qty for o in self._orders)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """현재가 기준 미실현 손익 (%)."""
        avg = self.avg_entry_price()
        if avg <= 0:
            return 0.0
        return (current_price - avg) / avg

    def safety_orders_remaining(self) -> int:
        """남은 Safety Order 횟수."""
        used = max(0, len(self._orders) - 1)  # 첫 매수 제외
        return max(0, self._max_safety - used)

    def reset(self) -> None:
        """포지션 완전 청산 후 초기화."""
        self._orders.clear()
        self._last_buy_price = 0.0
        logger.info("[DCA] 포지션 초기화")
