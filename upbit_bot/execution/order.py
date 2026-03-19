"""
execution/order.py — SmartOrderRouter + PartialFillHandler

업비트 주문 실행 계층.
- SmartOrderRouter: 호가창 유동성 체크 후 지정가/시장가 자동 선택
- PartialFillHandler: 부분 체결 감지 → 취소 후 시장가 Sweep
- DELETE 레이스 컨디션 예외 처리 (HTTP 400 "already done")
- DRY_RUN 모드: 실제 API 미호출, 시뮬레이션 결과 반환
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

TRADING_FEE_MAKER = 0.0005   # 지정가 maker 수수료 0.05%
TRADING_FEE_TAKER = 0.0005   # 시장가 taker 수수료 0.05%
SLIPPAGE_ESTIMATE  = 0.0015  # 슬리피지 추정 0.15%
TOTAL_COST_RATE    = TRADING_FEE_TAKER + SLIPPAGE_ESTIMATE  # 0.25%

# SmartOrderRouter 지정가 선택 조건
SPREAD_THRESHOLD          = 0.001   # spread < 0.1%
LIQUIDITY_MULTIPLIER      = 3.0     # 1호가 잔량 > 주문금액 × 3
TRADE_VELOCITY_THRESHOLD  = 2.0     # trade_velocity < 2.0

# PartialFillHandler 임계값
FILL_RATE_ACCEPT  = 0.80   # 체결률 ≥ 80% → 완료 처리
FILL_RATE_SWEEP   = 0.30   # 체결률 30~80% → 미체결 잔량 시장가 Sweep
FILL_WAIT_SEC     = 30     # 지정가 주문 후 대기 시간 (초)

# 재시도
MAX_ORDER_RETRY   = 3


class OrderResult(Enum):
    FULLY_FILLED     = "FULLY_FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED        = "CANCELLED"
    FAILED           = "FAILED"
    DRY_RUN          = "DRY_RUN"


class OrderType(Enum):
    LIMIT  = "limit"
    MARKET = "market"


@dataclass
class OrderRequest:
    """주문 요청 파라미터."""
    coin: str
    side: str           # 'BUY' | 'SELL'
    krw_amount: float   # KRW 기준 금액
    current_price: float
    strategy_type: str = ""
    is_emergency: bool = False      # 강제 손절 → SmartOrderRouter 우회, 즉시 시장가
    force_market: bool = False      # 시장가 강제


@dataclass
class OrderStatus:
    """업비트 주문 상태 (실거래/DRY_RUN 통합)."""
    order_id: str
    coin: str
    side: str
    order_type: OrderType
    requested_krw: float
    price: float                    # 지정가 (시장가면 체결 평균가)
    executed_volume: float = 0.0    # 체결 수량 (코인)
    executed_krw: float    = 0.0    # 체결 금액 (KRW)
    remaining_volume: float = 0.0   # 미체결 수량
    state: str = "wait"             # 'wait' | 'done' | 'cancel'
    result: OrderResult = OrderResult.DRY_RUN
    fee: float = 0.0
    slippage_pct: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def fill_rate(self) -> float:
        total = self.executed_volume + self.remaining_volume
        if total <= 0:
            return 0.0
        return self.executed_volume / total


class UpbitAPIError(Exception):
    """업비트 API 에러 래퍼."""
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


# ------------------------------------------------------------------
# UpbitClient — 실거래 API 래퍼 (DRY_RUN 모드 지원)
# ------------------------------------------------------------------

class UpbitClient:
    """pyupbit Upbit 클라이언트 래퍼.

    DRY_RUN=True 이면 실제 API를 호출하지 않고 시뮬레이션 결과를 반환한다.
    """

    def __init__(self, access_key: str = "", secret_key: str = "", dry_run: bool = True) -> None:
        self._dry_run = dry_run
        self._access = access_key
        self._secret = secret_key
        self._upbit: Any = None

        if not dry_run and access_key and secret_key:
            try:
                import pyupbit
                self._upbit = pyupbit.Upbit(access_key, secret_key)
                logger.info("[UpbitClient] 실거래 모드 초기화 완료")
            except ImportError:
                logger.warning("[UpbitClient] pyupbit 미설치 — DRY_RUN으로 전환")
                self._dry_run = True
        else:
            logger.info("[UpbitClient] DRY_RUN 모드")

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    async def get_orderbook(self, coin: str) -> dict[str, Any]:
        """호가창 조회."""
        if self._dry_run:
            # 시뮬레이션: 스프레드 0.05%, 충분한 유동성 가정
            return {
                "ask_price": 50_000_000.0,
                "bid_price": 49_975_000.0,
                "ask_size": 1.0,     # BTC 기준
                "bid_size": 1.0,
                "spread_pct": 0.0005,
            }
        # 실거래
        try:
            import pyupbit
            ob = pyupbit.get_orderbook(ticker=f"KRW-{coin}")
            if not ob:
                return {}
            units = ob[0]["orderbook_units"]
            best_ask = units[0]["ask_price"]
            best_bid = units[0]["bid_price"]
            spread = (best_ask - best_bid) / best_ask
            return {
                "ask_price": best_ask,
                "bid_price": best_bid,
                "ask_size": units[0]["ask_size"],
                "bid_size": units[0]["bid_size"],
                "spread_pct": spread,
            }
        except Exception as exc:
            raise UpbitAPIError(500, f"호가창 조회 실패: {exc}") from exc

    async def place_limit_order(
        self,
        coin: str,
        side: str,
        price: float,
        volume: float,
    ) -> str:
        """지정가 주문. 주문 ID 반환."""
        order_id = str(uuid.uuid4())
        if self._dry_run:
            logger.debug("[DRY_RUN] 지정가 %s %s price=%.0f vol=%.6f", side, coin, price, volume)
            return order_id

        try:
            ticker = f"KRW-{coin}"
            if side == "BUY":
                resp = self._upbit.buy_limit_order(ticker, price, volume)
            else:
                resp = self._upbit.sell_limit_order(ticker, price, volume)
            return resp.get("uuid", order_id)
        except Exception as exc:
            raise UpbitAPIError(500, str(exc)) from exc

    async def place_market_order(
        self,
        coin: str,
        side: str,
        krw_amount: float | None = None,
        volume: float | None = None,
    ) -> str:
        """시장가 주문. 주문 ID 반환."""
        order_id = str(uuid.uuid4())
        if self._dry_run:
            logger.debug("[DRY_RUN] 시장가 %s %s amount=%.0f", side, coin, krw_amount or 0)
            return order_id

        try:
            ticker = f"KRW-{coin}"
            if side == "BUY":
                resp = self._upbit.buy_market_order(ticker, krw_amount)
            else:
                resp = self._upbit.sell_market_order(ticker, volume)
            return resp.get("uuid", order_id)
        except Exception as exc:
            raise UpbitAPIError(500, str(exc)) from exc

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """주문 조회."""
        if self._dry_run:
            # DRY_RUN: 즉시 완전 체결 가정
            return {
                "uuid": order_id,
                "state": "done",
                "executed_volume": "0.001",
                "remaining_volume": "0.0",
                "price": "50000000.0",
                "avg_price": "50000000.0",
                "paid_fee": "25.0",
            }

        try:
            resp = self._upbit.get_order(order_id)
            return resp or {}
        except Exception as exc:
            raise UpbitAPIError(500, str(exc)) from exc

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """주문 취소. 이미 체결된 경우 UpbitAPIError(400, 'already done') 발생."""
        if self._dry_run:
            return {"uuid": order_id, "state": "cancel"}

        try:
            resp = self._upbit.cancel_order(order_id)
            return resp or {}
        except Exception as exc:
            msg = str(exc).lower()
            if "400" in msg or "already" in msg:
                raise UpbitAPIError(400, "already done") from exc
            raise UpbitAPIError(500, str(exc)) from exc

    async def get_balance(self, currency: str = "KRW") -> float:
        """잔고 조회."""
        if self._dry_run:
            return 10_000_000.0

        try:
            return float(self._upbit.get_balance(currency) or 0.0)
        except Exception as exc:
            raise UpbitAPIError(500, f"잔고 조회 실패: {exc}") from exc


# ------------------------------------------------------------------
# PartialFillHandler
# ------------------------------------------------------------------

class PartialFillHandler:
    """부분 체결 처리기.

    처리 절차:
      1. 지정가 주문 후 FILL_WAIT_SEC(30초) 대기
      2. 체결률 확인
         ≥ 80%: 미체결 잔량 취소 → 완료 처리
         30~80%: 기존 주문 취소 → 미체결 잔량만 시장가 Sweep
         < 30%: 기존 주문 취소 → 재진입 여부 외부 판단 (CANCELLED 반환)
      3. DELETE 레이스 컨디션: HTTP 400 "already done" → FULLY_FILLED 처리
    """

    def __init__(self, client: UpbitClient) -> None:
        self._client = client

    async def handle(
        self,
        order_id: str,
        coin: str,
        side: str,
        requested_krw: float,
        limit_price: float,
    ) -> OrderStatus:
        """부분 체결 처리 후 최종 OrderStatus 반환."""
        # Step 1: 대기
        await asyncio.sleep(FILL_WAIT_SEC if not self._client.dry_run else 0)

        # Step 2: 체결률 확인
        try:
            info = await self._client.get_order(order_id)
        except UpbitAPIError as exc:
            logger.error("[PartialFill] get_order 실패: %s", exc)
            raise

        executed_vol = float(info.get("executed_volume", 0))
        remaining_vol = float(info.get("remaining_volume", 0))
        avg_price = float(info.get("avg_price") or info.get("price") or limit_price)
        fee = float(info.get("paid_fee") or 0.0)

        total_vol = executed_vol + remaining_vol
        fill_rate = executed_vol / total_vol if total_vol > 0 else 0.0

        logger.info(
            "[PartialFill] %s %s fill_rate=%.1f%% exec=%.6f remain=%.6f",
            side, coin, fill_rate * 100, executed_vol, remaining_vol,
        )

        # ≥ 80%: 완료 처리
        if fill_rate >= FILL_RATE_ACCEPT:
            await self._safe_cancel(order_id)
            status = self._make_status(order_id, coin, side, requested_krw,
                                       avg_price, OrderResult.FULLY_FILLED)
            status.executed_volume = executed_vol
            status.executed_krw = executed_vol * avg_price
            status.fee = fee
            return status

        # 30~80%: 취소 후 미체결 잔량 시장가 Sweep
        if fill_rate >= FILL_RATE_SWEEP:
            fully_filled = await self._safe_cancel(order_id)
            if not fully_filled and remaining_vol > 0:
                # 미체결 잔량만큼 시장가 재발주
                sweep_krw = remaining_vol * avg_price if side == "BUY" else None
                sweep_vol = remaining_vol if side == "SELL" else None
                try:
                    await self._client.place_market_order(
                        coin, side, krw_amount=sweep_krw, volume=sweep_vol
                    )
                    logger.info("[PartialFill] 미체결 잔량 시장가 Sweep %.6f", remaining_vol)
                except UpbitAPIError as exc:
                    # 미체결 잔량 방치는 Kelly 포지션 사이징 오염 — 즉시 전파
                    raise

            status = self._make_status(order_id, coin, side, requested_krw,
                                       avg_price, OrderResult.PARTIALLY_FILLED)
            status.executed_volume = executed_vol
            status.executed_krw = executed_vol * avg_price
            status.fee = fee
            return status

        # < 30%: 취소 → 재진입 재평가 (CANCELLED)
        await self._safe_cancel(order_id)
        logger.warning("[PartialFill] 체결률 %.1f%% < 30%% — 재진입 재평가 필요", fill_rate * 100)
        status = self._make_status(order_id, coin, side, requested_krw,
                                   avg_price, OrderResult.CANCELLED)
        status.executed_volume = executed_vol
        return status

    async def _safe_cancel(self, order_id: str) -> bool:
        """취소 요청. 이미 체결된 경우(400 already done)를 FULLY_FILLED로 처리.

        Returns:
            True — 이미 100% 체결 완료 (레이스 컨디션)
            False — 정상 취소
        """
        try:
            await self._client.cancel_order(order_id)
            return False
        except UpbitAPIError as exc:
            if exc.code == 400 and "already done" in exc.message:
                logger.info("[PartialFill] 취소 찰나 100%% 체결 확인 (race condition): %s", order_id)
                return True  # FULLY_FILLED로 처리
            raise

    @staticmethod
    def _make_status(
        order_id: str,
        coin: str,
        side: str,
        requested_krw: float,
        price: float,
        result: OrderResult,
    ) -> OrderStatus:
        return OrderStatus(
            order_id=order_id,
            coin=coin,
            side=side,
            order_type=OrderType.LIMIT,
            requested_krw=requested_krw,
            price=price,
            result=result,
        )


# ------------------------------------------------------------------
# SmartOrderRouter
# ------------------------------------------------------------------

class SmartOrderRouter:
    """조건부 지정가/시장가 선택 + 부분 체결 처리.

    진입 조건 (3가지 모두 충족 시 지정가):
      - spread < 0.1%
      - 1호가 잔량 > 주문금액 × 3 (코인 수량 기준)
      - trade_velocity < 2.0

    조건 미충족(호가 얇음 / 급등) → 시장가.
    긴급 손절(is_emergency=True) → SmartOrderRouter 우회, 즉시 시장가.

    손절 분할 실행: split_emergency=True → 50%+50%, 3초 간격
    (서킷브레이커 Level 2 이상 전량 즉시는 split=False로 호출)
    """

    def __init__(self, client: UpbitClient) -> None:
        self._client = client
        self._partial_fill_handler = PartialFillHandler(client)

    async def execute(
        self,
        req: OrderRequest,
        trade_velocity: float = 1.0,
        split_emergency: bool = True,
    ) -> OrderStatus:
        """주문 실행 진입점.

        Args:
            req: 주문 요청
            trade_velocity: 체결 속도 가속도 (MarketState.trade_velocity)
            split_emergency: 긴급 손절 시 50%+50% 분할 여부

        Returns:
            OrderStatus (DRY_RUN 모드면 시뮬레이션 결과)
        """
        if req.krw_amount <= 0:
            raise ValueError(f"주문 금액이 0 이하: {req.krw_amount}")

        # 긴급 손절 → 즉시 시장가 (분할 실행 선택)
        if req.is_emergency:
            return await self._emergency_market(req, split=split_emergency)

        # 잔고 확인
        await self._check_balance(req)

        # 호가창 조회 → 주문 방식 결정
        ob = await self._client.get_orderbook(req.coin)
        use_limit = self._should_use_limit(ob, req.krw_amount, trade_velocity)

        if not req.force_market and use_limit:
            return await self._limit_order(req, ob)
        else:
            return await self._market_order(req)

    async def _emergency_market(
        self, req: OrderRequest, split: bool = True
    ) -> OrderStatus:
        """강제 손절: SmartOrderRouter 우회, 즉시 시장가.

        split=True → 50% + 50%, 3초 간격 (호가 충격 최소화).
        split=False → 전량 즉시.
        """
        logger.warning("[SmartOrderRouter] 긴급 손절 %s %s amount=%.0f split=%s",
                        req.side, req.coin, req.krw_amount, split)

        if not split:
            return await self._market_order(req)

        half = req.krw_amount / 2
        req1 = OrderRequest(
            coin=req.coin, side=req.side, krw_amount=half,
            current_price=req.current_price, strategy_type=req.strategy_type,
            is_emergency=False, force_market=True,
        )
        status1 = await self._market_order(req1)

        await asyncio.sleep(3)  # 3초 간격

        req2 = OrderRequest(
            coin=req.coin, side=req.side, krw_amount=half,
            current_price=req.current_price, strategy_type=req.strategy_type,
            is_emergency=False, force_market=True,
        )
        status2 = await self._market_order(req2)

        # 두 주문 합산
        status1.executed_volume += status2.executed_volume
        status1.executed_krw    += status2.executed_krw
        status1.fee             += status2.fee
        status1.requested_krw    = req.krw_amount
        return status1

    def _should_use_limit(
        self,
        ob: dict[str, Any],
        krw_amount: float,
        trade_velocity: float,
    ) -> bool:
        """지정가 선택 여부 판단."""
        if not ob:
            return False
        spread = ob.get("spread_pct", 1.0)
        ask_size = ob.get("ask_size", 0.0)   # BTC 수량
        ask_price = ob.get("ask_price", 1.0)

        # 주문금액 × 3 을 코인 수량으로 환산
        required_size = (krw_amount * LIQUIDITY_MULTIPLIER) / ask_price

        cond_spread   = spread < SPREAD_THRESHOLD
        cond_liquidity = ask_size >= required_size
        cond_velocity  = trade_velocity < TRADE_VELOCITY_THRESHOLD

        return cond_spread and cond_liquidity and cond_velocity

    async def _limit_order(
        self, req: OrderRequest, ob: dict[str, Any]
    ) -> OrderStatus:
        """지정가 주문 실행 → PartialFillHandler."""
        price = ob.get("ask_price" if req.side == "BUY" else "bid_price",
                        req.current_price)
        volume = req.krw_amount / price

        for attempt in range(1, MAX_ORDER_RETRY + 1):
            try:
                order_id = await self._client.place_limit_order(
                    req.coin, req.side, price, volume
                )
                logger.info(
                    "[SmartOrderRouter] 지정가 %s %s price=%.0f vol=%.6f id=%s",
                    req.side, req.coin, price, volume, order_id,
                )
                break
            except UpbitAPIError as exc:
                logger.warning("[SmartOrderRouter] 지정가 시도 %d 실패: %s", attempt, exc)
                if attempt == MAX_ORDER_RETRY:
                    raise
                await asyncio.sleep(2 ** (attempt - 1))  # 1s, 2s 지수 backoff

        # 부분 체결 처리
        return await self._partial_fill_handler.handle(
            order_id=order_id,
            coin=req.coin,
            side=req.side,
            requested_krw=req.krw_amount,
            limit_price=price,
        )

    async def _market_order(self, req: OrderRequest) -> OrderStatus:
        """시장가 주문 실행."""
        order_id = ""
        for attempt in range(1, MAX_ORDER_RETRY + 1):
            try:
                if req.side == "BUY":
                    order_id = await self._client.place_market_order(
                        req.coin, "BUY", krw_amount=req.krw_amount
                    )
                else:
                    if req.current_price <= 0:
                        raise ValueError(
                            f"current_price가 0 이하: {req.current_price} — SELL 수량 계산 불가"
                        )
                    volume = req.krw_amount / req.current_price
                    order_id = await self._client.place_market_order(
                        req.coin, "SELL", volume=volume
                    )
                logger.info(
                    "[SmartOrderRouter] 시장가 %s %s amount=%.0f id=%s",
                    req.side, req.coin, req.krw_amount, order_id,
                )
                break
            except UpbitAPIError as exc:
                logger.warning("[SmartOrderRouter] 시장가 시도 %d 실패: %s", attempt, exc)
                if attempt == MAX_ORDER_RETRY:
                    raise
                await asyncio.sleep(2 ** (attempt - 1))  # 1s, 2s 지수 backoff

        # 체결 정보 조회
        try:
            info = await self._client.get_order(order_id)
        except UpbitAPIError as exc:
            # 의도적 폴백: 시장가 주문은 이미 체결 완료이므로 취소 불가.
            # get_order 실패 시 current_price / krw_amount 기반 추정값으로 대체.
            # 이 경우 체결가·수량·수수료가 부정확하게 기록될 수 있으므로 error로 남긴다.
            logger.error("[SmartOrderRouter] 시장가 체결 정보 조회 실패 — 추정값 사용: %s", exc)
            info = {}

        executed_vol = float(info.get("executed_volume", req.krw_amount / max(req.current_price, 1)))
        avg_price    = float(info.get("avg_price") or req.current_price)
        fee          = float(info.get("paid_fee") or req.krw_amount * TRADING_FEE_TAKER)
        slippage     = abs(avg_price - req.current_price) / req.current_price if req.current_price else 0

        status = OrderStatus(
            order_id=order_id,
            coin=req.coin,
            side=req.side,
            order_type=OrderType.MARKET,
            requested_krw=req.krw_amount,
            price=avg_price,
            executed_volume=executed_vol,
            executed_krw=executed_vol * avg_price,
            state="done",
            result=OrderResult.FULLY_FILLED if not self._client.dry_run else OrderResult.DRY_RUN,
            fee=fee,
            slippage_pct=slippage,
        )
        return status

    async def _check_balance(self, req: OrderRequest) -> None:
        """잔고 부족 시 ValueError."""
        if self._client.dry_run:
            return
        if req.side == "BUY":
            balance = await self._client.get_balance("KRW")
            if balance < req.krw_amount:
                raise ValueError(
                    f"KRW 잔고 부족: 필요={req.krw_amount:.0f}, 보유={balance:.0f}"
                )
        else:
            balance = await self._client.get_balance(req.coin)
            required_vol = req.krw_amount / req.current_price if req.current_price else 0
            if balance < required_vol:
                raise ValueError(
                    f"{req.coin} 잔고 부족: 필요={required_vol:.6f}, 보유={balance:.6f}"
                )

    async def cancel_with_race_guard(self, order_id: str) -> OrderResult:
        """취소 요청 + 레이스 컨디션 처리.

        Returns:
            FULLY_FILLED — 이미 100% 체결 (레이스 컨디션)
            CANCELLED    — 정상 취소
            FAILED       — 예상치 못한 오류
        """
        try:
            await self._client.cancel_order(order_id)
            return OrderResult.CANCELLED
        except UpbitAPIError as exc:
            if exc.code == 400 and "already done" in exc.message:
                logger.info("[SmartOrderRouter] 취소 레이스 → FULLY_FILLED: %s", order_id)
                return OrderResult.FULLY_FILLED
            raise
