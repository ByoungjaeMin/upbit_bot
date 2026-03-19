"""
strategies/grid.py — GridStrategy (ATR 기반 횡보 그리드)

ATR×3 범위에 10단계 그리드 설정.
하단 5개: 지정가 매수 / 상단 5개: 지정가 매도.
체결 시 반대 레벨에 자동 재주문.
ADX > 25 → 추세 전환 → 그리드 해제.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ADX 임계값: 초과 시 추세 전환 → 그리드 종료
GRID_ADX_CLOSE_THRESHOLD = 25.0


@dataclass
class GridOrder:
    """단일 그리드 주문 상태."""
    level: int                         # 0~9 (0=최하단)
    price: float
    side: Literal["BUY", "SELL"]
    size_krw: float                    # KRW 주문금액
    filled: bool = False
    order_id: str = ""


class GridStrategy:
    """ATR 기반 그리드 전략.

    사용법:
        grid = GridStrategy(capital=200_000, current_price=50_000_000, atr=1_500_000)
        orders = grid.place_grid_orders()
        # 체결 통보
        reorder = grid.on_order_filled(level=2, side='BUY')
    """

    def __init__(
        self,
        capital: float,           # 투입 KRW 자본
        current_price: float,     # 현재가 (KRW)
        atr: float,               # ATR 값 (KRW)
        levels: int = 10,         # 그리드 단계 수
        atr_multiplier: float = 3.0,
    ) -> None:
        if current_price <= 0 or atr <= 0:
            raise ValueError(f"current_price, atr는 양수여야 함: {current_price}, {atr}")

        self._capital = capital
        self._levels = levels
        self._atr_multiplier = atr_multiplier

        # 그리드 범위
        self.range_upper = current_price + atr * atr_multiplier
        self.range_lower = max(current_price - atr * atr_multiplier, 0.0)
        self.current_price = current_price

        # 그리드 가격 배열 (levels개 균등 분할)
        self.grid_prices: np.ndarray = np.linspace(
            self.range_lower, self.range_upper, levels
        )

        # 레벨당 KRW 금액
        self.unit_size: float = capital / levels

        # 주문 상태 저장소 (level → GridOrder)
        self._orders: dict[int, GridOrder] = {}

        logger.info(
            "[Grid] 초기화 lower=%.0f upper=%.0f unit=%.0f KRW",
            self.range_lower, self.range_upper, self.unit_size,
        )

    # ------------------------------------------------------------------
    # 그리드 주문 설정
    # ------------------------------------------------------------------

    def place_grid_orders(self) -> list[GridOrder]:
        """하단 절반 → 지정가 매수 / 상단 절반 → 지정가 매도.

        Returns:
            설정된 GridOrder 목록 (order_id는 빈 문자열, 실제 주문 후 채워야 함)
        """
        orders: list[GridOrder] = []
        mid = self._levels // 2

        for i, price in enumerate(self.grid_prices):
            side: Literal["BUY", "SELL"] = "BUY" if i < mid else "SELL"
            order = GridOrder(
                level=i,
                price=float(price),
                side=side,
                size_krw=self.unit_size,
            )
            self._orders[i] = order
            orders.append(order)

        logger.info(
            "[Grid] %d개 주문 설정 (BUY=%d, SELL=%d)",
            len(orders), mid, self._levels - mid,
        )
        return orders

    # ------------------------------------------------------------------
    # 체결 처리 → 반대 레벨 재주문
    # ------------------------------------------------------------------

    def on_order_filled(
        self, level: int, side: Literal["BUY", "SELL"]
    ) -> GridOrder | None:
        """체결 시 반대 방향 레벨에 재주문.

        BUY 체결 → 한 레벨 위에 SELL 재주문.
        SELL 체결 → 한 레벨 아래에 BUY 재주문.

        Returns:
            재주문 GridOrder (범위 초과 시 None)
        """
        if level in self._orders:
            self._orders[level].filled = True

        if side == "BUY":
            new_level = level + 1
            new_side: Literal["BUY", "SELL"] = "SELL"
        else:
            new_level = level - 1
            new_side = "BUY"

        if not (0 <= new_level < self._levels):
            logger.debug("[Grid] 재주문 범위 초과 (level=%d)", new_level)
            return None

        new_order = GridOrder(
            level=new_level,
            price=float(self.grid_prices[new_level]),
            side=new_side,
            size_krw=self.unit_size,
        )
        self._orders[new_level] = new_order
        logger.info(
            "[Grid] 재주문 level=%d side=%s price=%.0f",
            new_level, new_side, new_order.price,
        )
        return new_order

    # ------------------------------------------------------------------
    # 그리드 종료 조건
    # ------------------------------------------------------------------

    def should_close(self, adx: float) -> bool:
        """ADX > threshold → 추세 전환 → 그리드 해제."""
        return adx > GRID_ADX_CLOSE_THRESHOLD

    # ------------------------------------------------------------------
    # 범위 재계산 (4시간마다 APScheduler 호출)
    # ------------------------------------------------------------------

    def recalculate_range(self, current_price: float, atr: float) -> None:
        """ATR 기반 그리드 범위 재계산 + 주문 목록 초기화."""
        self.current_price = current_price
        self.range_upper = current_price + atr * self._atr_multiplier
        self.range_lower = max(current_price - atr * self._atr_multiplier, 0.0)
        self.grid_prices = np.linspace(self.range_lower, self.range_upper, self._levels)
        self._orders.clear()
        logger.info(
            "[Grid] 범위 재계산 lower=%.0f upper=%.0f",
            self.range_lower, self.range_upper,
        )

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------

    def get_open_orders(self) -> list[GridOrder]:
        return [o for o in self._orders.values() if not o.filled]

    def get_filled_count(self) -> int:
        return sum(1 for o in self._orders.values() if o.filled)

    @property
    def is_price_in_range(self) -> bool:
        return self.range_lower <= self.current_price <= self.range_upper
