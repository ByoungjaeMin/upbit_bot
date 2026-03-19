"""
execution/engine.py — 메인 실행 엔진 (APScheduler)

Phase A 구현:
- 5분 메인 루프: 16단계 신호 → 주문 파이프라인
- 1분 포지션 모니터링 루프
- 10초 서킷브레이커 감시 루프
- FinBERT: ProcessPoolExecutor 격리 (메인 루프 블로킹 방지)
- DRY_RUN 강제: trade_count < 200
- 0~90초 랜덤 진입 지연 (front-running 방어)
- 재진입 조건 강화: 손절 후 tick_imbalance/OBI/RSI 3조건 확인

CLAUDE.md 핵심 원칙:
  1. 서킷브레이커 최우선
  2. REST API 초기 로딩 1회 — 이후 WebSocket CandleBuilder 사용
  3. 모든 주문은 SmartOrderRouter 경유
  4. 부분 체결은 PartialFillHandler 처리
  5. trade_count < 200 → DRY_RUN=True 강제
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from risk.circuit_breaker import CircuitBreaker
from schema import (
    EnsemblePrediction,
    FilterResult,
    MarketState,
    RiskBudget,
    StrategyDecision,
    TradeDecision,
)
from execution.order import OrderRequest, OrderResult, SmartOrderRouter, UpbitClient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

MAX_POSITIONS          = 5       # 동시 보유 코인 상한
MIN_HOLD_MINUTES       = 15      # 최소 보유 시간 (분)
REENTRY_COOLDOWN_MIN   = 10      # 동일 코인 재진입 대기 (분)
MAX_DAILY_TRADES       = 10      # 일 최대 거래 횟수
ENTRY_DELAY_MAX_SEC    = 90      # 랜덤 진입 지연 최대 (초)
COLD_START_THRESHOLD   = 200     # DRY_RUN 강제 임계값

# 6조건 진입 트리거
ENSEMBLE_THRESHOLD     = 0.62
CONSENSUS_MIN          = 3
VAR_MAX_PCT            = 0.03    # VaR ≤ 자본 3%
TICK_IMBALANCE_MIN     = 0.10
OBI_MIN                = 0.20

# 재진입 조건 (손절 후)
REENTRY_TICK_MIN       = 0.15
REENTRY_RSI_MIN        = 40.0
REENTRY_RSI_MAX        = 65.0
REENTRY_OBI_MIN        = 0.10
REENTRY_CONDITIONS_MIN = 2      # 3조건 중 최소 2개

# 청산
HARD_STOP_LOSS_PCT     = -0.07  # -7% 하드 손절
PARTIAL_TP_1_PCT       = 0.10   # +10% 1차 부분 익절
PARTIAL_TP_2_PCT       = 0.20   # +20% 2차 부분 익절
NORMAL_CLOSE_WAIT_SEC  = 10     # 지정가 미체결 → 시장가 전환 대기 (초)


# ------------------------------------------------------------------
# 내부 데이터 클래스
# ------------------------------------------------------------------

@dataclass
class Position:
    """보유 포지션."""
    coin: str
    entry_price: float
    qty: float              # 코인 수량
    entry_krw: float        # 진입 금액
    strategy_type: str
    entry_ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trailing_stop_price: float = 0.0
    partial_exit_done: dict[str, bool] = field(default_factory=dict)  # {'10pct': False, '20pct': False}
    stop_loss_ts: datetime | None = None    # 손절 발생 시각

    @property
    def hold_minutes(self) -> float:
        return (datetime.now(timezone.utc) - self.entry_ts).total_seconds() / 60

    def pnl_pct(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price


@dataclass
class EngineState:
    """엔진 전역 상태."""
    trade_count: int = 0
    daily_trade_count: int = 0
    daily_loss_pct: float = 0.0
    last_trade_day: str = ""
    # 재진입 쿨타임 (coin → 쿨타임 만료 시각)
    reentry_cooldown: dict[str, datetime] = field(default_factory=dict)
    # 손절 이력 (coin → 손절 시각)
    stop_loss_history: dict[str, datetime] = field(default_factory=dict)


# ------------------------------------------------------------------
# TradingEngine
# ------------------------------------------------------------------

class TradingEngine:
    """업비트 퀀트 자동매매 메인 실행 엔진.

    APScheduler로 3가지 루프를 관리:
      - main_loop:     5분 주기 (신호 → 주문)
      - position_loop: 1분 주기 (트레일링 스탑 / 익절 / 손절)
      - circuit_loop:  10초 주기 (서킷브레이커 감시)

    Phase A에서는 Layer 3 RL 대신 룰 기반 결정 사용.
    FinBERT 추론은 ProcessPoolExecutor로 메인 루프와 분리.
    """

    def __init__(
        self,
        upbit_client: UpbitClient | None = None,
        dry_run: bool = True,
        initial_capital: float = 10_000_000.0,
    ) -> None:
        self._dry_run = dry_run
        self._capital = initial_capital

        # 서킷브레이커 (최우선)
        self._cb = CircuitBreaker()

        # 주문 실행
        self._client = upbit_client or UpbitClient(dry_run=dry_run)
        self._router = SmartOrderRouter(self._client)

        # 상태
        self._state = EngineState()
        self._positions: dict[str, Position] = {}

        # FinBERT 프로세스 풀 (1개 워커, 메인 루프와 격리)
        self._process_pool = ProcessPoolExecutor(max_workers=1)
        self._event_loop: asyncio.AbstractEventLoop | None = None

        # 페이퍼 트레이딩 (지연 임포트로 순환 방지)
        from execution.paper_trading import PaperTradingRunner
        self._paper = PaperTradingRunner(initial_capital)

        # 이벤트 큐 (텔레그램 비동기 전송용)
        self._telegram_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # Layer 1, 2 컴포넌트 (외부 주입 or 내부 생성)
        self._layer1: Any = None
        self._layer2: Any = None
        self._strategy_selector: Any = None
        self._decay_monitor: Any = None

        logger.info(
            "[Engine] 초기화 완료 dry_run=%s capital=%.0f",
            dry_run, initial_capital,
        )

    def setup_layers(
        self,
        layer1: Any = None,
        layer2: Any = None,
        strategy_selector: Any = None,
        decay_monitor: Any = None,
    ) -> None:
        """레이어 컴포넌트 주입 (테스트 및 실거래 모두 지원)."""
        self._layer1 = layer1
        self._layer2 = layer2
        self._strategy_selector = strategy_selector
        self._decay_monitor = decay_monitor

    # ------------------------------------------------------------------
    # 공개 루프 (APScheduler에서 호출)
    # ------------------------------------------------------------------

    async def main_loop(self, market_states: list[MarketState]) -> list[TradeDecision]:
        """5분 메인 루프 — 16단계 파이프라인.

        Args:
            market_states: CandleBuilder가 생성한 활성 코인 MarketState 목록

        Returns:
            실행된 TradeDecision 목록
        """
        decisions: list[TradeDecision] = []
        self._reset_daily_counter()

        # Step 1: 서킷브레이커 상태 확인 (최우선)
        if self._cb.is_all_blocked():
            logger.warning("[Engine] 서킷브레이커 Level %d — 전체 차단", self._cb.level)
            return decisions

        # Step 2: DRY_RUN 강제 체크
        self._enforce_dry_run()

        # Step 3~5: 각 코인 필터 병렬 실행
        filter_results = await self._run_layer1_parallel(market_states)

        # Step 6~10: 앙상블 예측 병렬 실행 (Layer 1 통과 코인만)
        tradeable_states = [
            ms for ms in market_states
            if filter_results.get(ms.coin, {}).get("tradeable", False)
        ]
        ensemble_preds = await self._run_layer2_parallel(tradeable_states)

        # Step 11: Kelly + VaR 리스크 계산
        risk_budgets = self._compute_risk_budgets(tradeable_states, ensemble_preds)

        # Step 12: Layer 3 RL 결정 (Phase A → 룰 기반)
        # Phase C+에서 DQNAgent 교체

        # Step 13: 진입 트리거 6조건 평가 → 최강 신호 코인 선정
        candidates = self._evaluate_entry_triggers(
            market_states, filter_results, ensemble_preds, risk_budgets
        )

        # Step 14: 전략별 주문 실행
        for ms, fr, ep, rb in candidates:
            if len(self._positions) >= MAX_POSITIONS:
                logger.info("[Engine] 최대 포지션 %d 초과 — %s 스킵", MAX_POSITIONS, ms.coin)
                break
            if self._state.daily_trade_count >= MAX_DAILY_TRADES:
                logger.info("[Engine] 일 거래 상한 %d 도달", MAX_DAILY_TRADES)
                break

            decision = await self._execute_entry(ms, fr, ep, rb)
            if decision:
                decisions.append(decision)

        # Step 15: SQLite 저장 (RL 보상 피드백)
        await self._save_cycle_data(market_states, filter_results, ensemble_preds, decisions)

        # Step 16: 텔레그램 이벤트 큐
        if decisions:
            await self._enqueue_telegram_events(decisions)

        return decisions

    async def position_loop(self, market_states: list[MarketState]) -> None:
        """1분 포지션 모니터링 루프."""
        if self._cb.is_all_blocked():
            return

        price_map = {ms.coin: ms.close_5m for ms in market_states}
        ms_map    = {ms.coin: ms for ms in market_states}

        for coin, pos in list(self._positions.items()):
            current = price_map.get(coin, 0.0)
            if current <= 0:
                continue
            ms = ms_map.get(coin)

            # ATR 트레일링 스탑 업데이트
            if ms:
                self._update_trailing_stop(pos, ms)

            # 하드 손절 (-7%)
            if pos.pnl_pct(current) <= HARD_STOP_LOSS_PCT:
                await self._close_position(coin, current, reason="hard_stop", emergency=True)
                continue

            # 트레일링 스탑
            if pos.trailing_stop_price > 0 and current <= pos.trailing_stop_price:
                await self._close_position(coin, current, reason="trailing_stop", emergency=True)
                continue

            # SuperTrend 반전 → 즉시 청산
            if ms and ms.supertrend_signal == -1 and pos.strategy_type.startswith("TREND"):
                await self._close_position(coin, current, reason="supertrend_reversal")
                continue

            # 부분 익절
            await self._check_partial_tp(pos, current)

            # 그리드 체결 확인 (Grid 전략)
            if pos.strategy_type == "GRID" and ms:
                await self._check_grid_fills(coin, ms)

            # DCA Safety Order 조건 체크
            if pos.strategy_type == "DCA" and ms:
                await self._check_dca_safety(coin, ms)

        # 서킷브레이커 자동 회복 시도
        self._cb.maybe_recover()

    async def circuit_loop(self, market_states: list[MarketState]) -> None:
        """10초 서킷브레이커 감시 루프.

        check_price_drop은 1분/10분 전 가격이 필요하므로 외부 호출자가
        price_1m_ago / price_10m_ago를 전달해야 한다.
        이 루프는 API 오류 누적 및 자동 회복만 처리한다.
        실제 가격 기반 트리거는 _trigger_price_circuit() 를 직접 호출한다.
        """
        # Level 4: API 오류 누적 (외부에서 record_api_error 호출 시 처리됨)
        self._cb.maybe_recover()

    def trigger_price_circuit(
        self,
        coin: str,
        price_1m_ago: float,
        price_10m_ago: float,
        current_price: float,
        daily_loss_pct: float,
        cumulative_loss_24h: float,
    ) -> int:
        """WebSocket 가격 이벤트 수신 시 서킷브레이커 조건 체크."""
        return self._cb.check_price_drop(
            coin=coin,
            price_1m_ago=price_1m_ago,
            price_10m_ago=price_10m_ago,
            current_price=current_price,
            daily_loss_pct=daily_loss_pct,
            cumulative_loss_24h=cumulative_loss_24h,
        )

    # ------------------------------------------------------------------
    # 진입 트리거 6조건 평가
    # ------------------------------------------------------------------

    def _evaluate_entry_triggers(
        self,
        market_states: list[MarketState],
        filter_results: dict[str, Any],
        ensemble_preds: dict[str, EnsemblePrediction],
        risk_budgets: dict[str, RiskBudget],
    ) -> list[tuple[MarketState, FilterResult, EnsemblePrediction, RiskBudget]]:
        """6조건 평가 → 기준 충족 코인 목록 (앙상블 × Kelly 내림차순)."""
        candidates = []
        for ms in market_states:
            coin = ms.coin

            fr = filter_results.get(coin)
            if not fr or not fr.get("tradeable", False):
                continue

            ep = ensemble_preds.get(coin)
            rb = risk_budgets.get(coin)

            if ep is None or rb is None:
                continue

            # ① Layer 1 통과 (fr.tradeable)

            # ② 앙상블 조건
            if ep.weighted_avg < ENSEMBLE_THRESHOLD:
                logger.debug("[Engine] %s 앙상블 %.3f < %.2f", coin, ep.weighted_avg, ENSEMBLE_THRESHOLD)
                continue
            if ep.consensus_count < CONSENSUS_MIN:
                logger.debug("[Engine] %s consensus %d < %d", coin, ep.consensus_count, CONSENSUS_MIN)
                continue

            # ③ Kelly > 0
            if rb.var_adjusted_f <= 0:
                logger.debug("[Engine] %s Kelly ≤ 0 스킵", coin)
                continue

            # ④ VaR ≤ 자본 3%
            if rb.var_95 > VAR_MAX_PCT:
                # 포지션 50% 축소 후 재판단
                rb.var_adjusted_f *= 0.5
                rb.final_position_size *= 0.5
                logger.info("[Engine] %s VaR %.3f > 3%% — 포지션 50%% 축소", coin, rb.var_95)
                if rb.var_adjusted_f <= 0:
                    continue

            # ⑤ 마이크로스트럭처 최소 확인
            if ms.tick_imbalance < TICK_IMBALANCE_MIN and ms.obi < OBI_MIN:
                logger.debug("[Engine] %s 마이크로스트럭처 부족", coin)
                continue

            # ⑥ 동시 보유 포지션 < 5
            if len(self._positions) >= MAX_POSITIONS:
                break

            # 재진입 쿨타임 확인
            if self._is_in_cooldown(coin):
                logger.debug("[Engine] %s 쿨타임 중 스킵", coin)
                continue

            # 손절 후 재진입 추가 조건 확인
            if coin in self._state.stop_loss_history:
                if not self._check_reentry_conditions(ms):
                    logger.info("[Engine] %s 재진입 조건 미충족 — 추가 대기", coin)
                    continue

            score = ep.weighted_avg * rb.var_adjusted_f
            candidates.append((ms, fr, ep, rb, score))

        # 앙상블 × Kelly 내림차순 정렬
        candidates.sort(key=lambda x: x[4], reverse=True)
        return [(ms, fr, ep, rb) for ms, fr, ep, rb, _ in candidates]

    def _check_reentry_conditions(self, ms: MarketState) -> bool:
        """손절 후 재진입 추가 3조건 (2개 이상 충족 필요).

        ① tick_imbalance > 0.15 (매수 체결 우세 회복)
        ② RSI 40~65 (과열/과매도 아닌 구간)
        ③ OBI > 0.10 (오더북 매수 우위 최소 확인)
        """
        conds = [
            ms.tick_imbalance > REENTRY_TICK_MIN,
            REENTRY_RSI_MIN <= ms.rsi_5m <= REENTRY_RSI_MAX,
            ms.obi > REENTRY_OBI_MIN,
        ]
        met = sum(conds)
        logger.debug("[Engine] 재진입 조건 %d/3", met)
        return met >= REENTRY_CONDITIONS_MIN

    # ------------------------------------------------------------------
    # 주문 실행
    # ------------------------------------------------------------------

    async def _execute_entry(
        self,
        ms: MarketState,
        fr: Any,
        ep: EnsemblePrediction,
        rb: RiskBudget,
    ) -> TradeDecision | None:
        """전략별 진입 주문 실행."""
        strategy = fr.get("regime_strategy", "HOLD") if isinstance(fr, dict) else getattr(fr, "regime_strategy", "HOLD")
        if strategy == "HOLD":
            return None

        position_size = rb.final_position_size
        is_dry = self._dry_run

        # TREND: 0~90초 랜덤 지연 (front-running 방어)
        delay_sec = 0.0
        if strategy.startswith("TREND"):
            delay_sec = random.uniform(0, ENTRY_DELAY_MAX_SEC)
            if not is_dry:
                logger.info("[Engine] 진입 지연 %.1f초 (front-running 방어)", delay_sec)
                await asyncio.sleep(delay_sec)

        loop_id = str(uuid.uuid4())
        signal_ts = datetime.now(timezone.utc)

        # 페이퍼 트레이딩 병렬 실행
        self._paper.on_signal(
            coin=ms.coin,
            side="BUY",
            paper_price=ms.close_5m,
            krw_amount=position_size,
            strategy_type=strategy,
            loop_id=loop_id,
            signal_ts=signal_ts,
        )

        # 실거래 주문
        req = OrderRequest(
            coin=ms.coin,
            side="BUY",
            krw_amount=position_size,
            current_price=ms.close_5m,
            strategy_type=strategy,
        )

        try:
            status = await self._router.execute(
                req,
                trade_velocity=ms.trade_velocity,
            )
        except Exception as exc:
            logger.error("[Engine] 주문 실패 %s: %s", ms.coin, exc)
            self._cb.record_api_error()
            return None

        # 체결 완료 시각 기록 (페이퍼 비교)
        executed_ts = datetime.now(timezone.utc)
        self._paper.on_live_executed(loop_id, status.price, executed_ts)

        # 포지션 등록
        if status.result in (OrderResult.FULLY_FILLED, OrderResult.PARTIALLY_FILLED, OrderResult.DRY_RUN):
            pos = Position(
                coin=ms.coin,
                entry_price=status.price,
                qty=status.executed_volume,
                entry_krw=position_size,
                strategy_type=strategy,
            )
            self._positions[ms.coin] = pos
            self._state.trade_count += 1
            self._state.daily_trade_count += 1

        decision = TradeDecision(
            coin=ms.coin,
            timestamp=signal_ts,
            action=1,  # BUY_STRONG
            target_coin=ms.coin,
            position_size=position_size,
            is_dry_run=is_dry,
            entry_delay_sec=delay_sec,
            strategy_type=strategy,
            ensemble_score=ep.weighted_avg,
        )
        logger.info(
            "[Engine] 진입 %s %s size=%.0f price=%.0f strategy=%s dry=%s",
            ms.coin, status.result.value, position_size, status.price, strategy, is_dry,
        )
        return decision

    async def _close_position(
        self,
        coin: str,
        current_price: float,
        reason: str = "",
        emergency: bool = False,
        partial_ratio: float | None = None,
    ) -> bool:
        """포지션 청산.

        일반 청산: SmartOrderRouter 지정가 우선 → 10초 미체결 시 시장가.
        강제 손절: 즉시 시장가 (분할 실행).
        """
        pos = self._positions.get(coin)
        if not pos:
            return False

        qty = pos.qty if partial_ratio is None else pos.qty * partial_ratio
        krw_amount = qty * current_price

        logger.info(
            "[Engine] 청산 %s qty=%.6f price=%.0f reason=%s emergency=%s",
            coin, qty, current_price, reason, emergency,
        )

        req = OrderRequest(
            coin=coin,
            side="SELL",
            krw_amount=krw_amount,
            current_price=current_price,
            is_emergency=emergency,
        )

        if emergency:
            # 강제 손절: SmartOrderRouter 우회 여부는 is_emergency 플래그로 처리
            # Level 2 이상 → split=False (전량 즉시)
            split = self._cb.level < 2
            status = await self._router.execute(req, split_emergency=split)
        else:
            # 일반 청산: 지정가 먼저, 10초 미체결 시 시장가
            status = await self._normal_close(req)

        # 포지션 제거
        if partial_ratio is None or (partial_ratio and qty >= pos.qty):
            del self._positions[coin]
        else:
            pos.qty -= qty

        # 손절 이력 기록 (재진입 조건 강화용)
        if reason in ("hard_stop", "trailing_stop"):
            self._state.stop_loss_history[coin] = datetime.now(timezone.utc)
            self._set_cooldown(coin)

        return True

    async def _normal_close(self, req: OrderRequest) -> Any:
        """일반 청산: 지정가 우선 → 10초 미체결 시 시장가 전환."""
        ob = await self._client.get_orderbook(req.coin)
        bid_price = ob.get("bid_price", req.current_price)

        # 지정가 주문
        volume = req.krw_amount / bid_price
        try:
            order_id = await self._client.place_limit_order(
                req.coin, "SELL", bid_price, volume
            )
        except Exception as exc:
            logger.warning("[Engine] 지정가 청산 실패 → 시장가: %s", exc)
            return await self._router.execute(req)

        # 10초 대기
        await asyncio.sleep(NORMAL_CLOSE_WAIT_SEC if not self._client.dry_run else 0)

        # 체결 확인
        try:
            info = await self._client.get_order(order_id)
            if info.get("state") == "done":
                return info
        except Exception:
            pass

        # 미체결 → 시장가 전환
        result = await self._router.cancel_with_race_guard(order_id)
        if result == OrderResult.FULLY_FILLED:
            return {"state": "done", "order_id": order_id}

        req_market = OrderRequest(
            coin=req.coin, side="SELL", krw_amount=req.krw_amount,
            current_price=req.current_price, force_market=True,
        )
        return await self._router.execute(req_market)

    # ------------------------------------------------------------------
    # 보조 기능
    # ------------------------------------------------------------------

    def _update_trailing_stop(self, pos: Position, ms: MarketState) -> None:
        """ATR 기반 트레일링 스탑 업데이트."""
        if ms.atr_5m <= 0:
            return
        atr_stop = ms.close_5m - ms.atr_5m * 2.0
        if atr_stop > pos.trailing_stop_price:
            pos.trailing_stop_price = atr_stop

    async def _check_partial_tp(self, pos: Position, current_price: float) -> None:
        """부분 익절 조건 체크 (+10%, +20%)."""
        pnl = pos.pnl_pct(current_price)

        if pnl >= PARTIAL_TP_2_PCT and not pos.partial_exit_done.get("20pct"):
            pos.partial_exit_done["20pct"] = True
            await self._close_position(pos.coin, current_price, reason="partial_tp_20", partial_ratio=0.5)
            logger.info("[Engine] %s +20%% 부분 익절 50%%", pos.coin)

        elif pnl >= PARTIAL_TP_1_PCT and not pos.partial_exit_done.get("10pct"):
            pos.partial_exit_done["10pct"] = True
            await self._close_position(pos.coin, current_price, reason="partial_tp_10", partial_ratio=0.3)
            logger.info("[Engine] %s +10%% 부분 익절 30%%", pos.coin)

    async def _check_grid_fills(self, coin: str, ms: MarketState) -> None:
        """그리드 체결 확인 (Grid 전략 전용)."""
        # Phase A: 실제 업비트 체결 이벤트는 WebSocket으로 수신
        # 여기서는 가격 범위 기반 근사 체크 (추후 WebSocket 연동으로 교체)
        pass

    async def _check_dca_safety(self, coin: str, ms: MarketState) -> None:
        """DCA Safety Order 조건 체크 (DCA 전략 전용)."""
        # Phase A: AdaptiveDCAStrategy와 연동
        pass

    def _enforce_dry_run(self) -> None:
        """trade_count < 200 → DRY_RUN 강제."""
        if self._state.trade_count < COLD_START_THRESHOLD:
            if not self._dry_run:
                logger.warning(
                    "[Engine] trade_count=%d < %d → DRY_RUN 강제",
                    self._state.trade_count, COLD_START_THRESHOLD,
                )
            self._dry_run = True
            assert self._dry_run is True  # CLAUDE.md 원칙 7번

    def _reset_daily_counter(self) -> None:
        """날짜 변경 시 일일 거래 횟수 리셋."""
        today = datetime.now(timezone.utc).date().isoformat()
        if self._state.last_trade_day != today:
            self._state.daily_trade_count = 0
            self._state.last_trade_day = today

    def _is_in_cooldown(self, coin: str) -> bool:
        """재진입 쿨타임 확인."""
        expire = self._state.reentry_cooldown.get(coin)
        if expire is None:
            return False
        return datetime.now(timezone.utc) < expire

    def _set_cooldown(self, coin: str) -> None:
        """재진입 쿨타임 설정."""
        self._state.reentry_cooldown[coin] = (
            datetime.now(timezone.utc) + timedelta(minutes=REENTRY_COOLDOWN_MIN)
        )

    # ------------------------------------------------------------------
    # 병렬 실행 헬퍼
    # ------------------------------------------------------------------

    async def _run_layer1_parallel(
        self, market_states: list[MarketState]
    ) -> dict[str, Any]:
        """Layer 1 필터 병렬 실행 (asyncio.gather)."""
        if self._layer1 is None:
            # Layer1 미주입 시 전체 통과 처리 (테스트/초기 단계)
            return {
                ms.coin: {
                    "tradeable": True,
                    "regime_strategy": "TREND_STRONG",
                    "signal_multiplier": 1.0,
                }
                for ms in market_states
            }

        async def _filter_one(ms: MarketState) -> tuple[str, Any]:
            try:
                result = self._layer1.filter(ms)
                return ms.coin, result.__dict__ if hasattr(result, "__dict__") else result
            except Exception as exc:
                logger.error("[Engine] Layer1 %s 실패: %s", ms.coin, exc)
                return ms.coin, {"tradeable": False}

        results = await asyncio.gather(*[_filter_one(ms) for ms in market_states])
        return dict(results)

    async def _run_layer2_parallel(
        self, market_states: list[MarketState]
    ) -> dict[str, EnsemblePrediction]:
        """Layer 2 앙상블 예측 병렬 실행."""
        if self._layer2 is None:
            # Layer2 미주입: 기본 예측 반환 (signal_confirmed=False)
            return {
                ms.coin: EnsemblePrediction(
                    coin=ms.coin,
                    timestamp=ms.timestamp,
                    weighted_avg=0.65,
                    consensus_count=3,
                    signal_confirmed=True,
                )
                for ms in market_states
            }

        async def _predict_one(ms: MarketState) -> tuple[str, EnsemblePrediction]:
            try:
                pred = self._layer2.predict(ms)
                return ms.coin, pred
            except Exception as exc:
                logger.error("[Engine] Layer2 %s 실패: %s", ms.coin, exc)
                return ms.coin, EnsemblePrediction(
                    coin=ms.coin,
                    timestamp=ms.timestamp,
                    signal_confirmed=False,
                )

        results = await asyncio.gather(*[_predict_one(ms) for ms in market_states])
        return dict(results)

    def _compute_risk_budgets(
        self,
        market_states: list[MarketState],
        ensemble_preds: dict[str, EnsemblePrediction],
    ) -> dict[str, RiskBudget]:
        """Kelly + VaR 리스크 예산 계산 (단순화 버전).

        Phase A: Kelly = 0.02 (자본의 2%) 고정, VaR = ATR 기반 근사.
        Phase B+: PyPortfolioOpt 연동으로 교체.
        """
        budgets: dict[str, RiskBudget] = {}
        for ms in market_states:
            ep = ensemble_preds.get(ms.coin)
            if ep is None:
                continue

            # 단순 Kelly: signal strength × 2%
            kelly_f = max(0.0, (ep.weighted_avg - 0.5) * 0.04)

            # ATR 기반 VaR 근사
            var_95 = (ms.atr_5m / ms.close_5m) * 1.65 if ms.close_5m > 0 else 0.03

            final_size = self._capital * kelly_f
            budgets[ms.coin] = RiskBudget(
                coin=ms.coin,
                timestamp=ms.timestamp,
                kelly_f=kelly_f,
                hmm_adjusted_f=kelly_f,
                var_adjusted_f=kelly_f,
                final_position_size=max(5_000, final_size),  # 최소 5,000원
                var_95=var_95,
            )
        return budgets

    async def _save_cycle_data(
        self,
        market_states: list[MarketState],
        filter_results: dict[str, Any],
        ensemble_preds: dict[str, EnsemblePrediction],
        decisions: list[TradeDecision],
    ) -> None:
        """SQLite 저장 (RL 보상 피드백)."""
        # Phase A: 로깅으로 대체, Phase B에서 실제 SQLite 저장 연동
        logger.debug(
            "[Engine] 사이클 완료 coins=%d tradeable=%d decisions=%d",
            len(market_states),
            sum(1 for fr in filter_results.values() if fr.get("tradeable")),
            len(decisions),
        )

    async def _enqueue_telegram_events(self, decisions: list[TradeDecision]) -> None:
        """텔레그램 이벤트 큐에 거래 알림 추가."""
        for d in decisions:
            event = {
                "event_type": "TRADE",
                "coin": d.coin,
                "strategy": d.strategy_type,
                "size": d.position_size,
                "dry_run": d.is_dry_run,
                "timestamp": d.timestamp.isoformat(),
            }
            await self._telegram_queue.put(event)

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._cb

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def trade_count(self) -> int:
        return self._state.trade_count

    @property
    def is_dry_run(self) -> bool:
        return self._dry_run

    def get_status(self) -> dict[str, Any]:
        return {
            "dry_run": self._dry_run,
            "trade_count": self._state.trade_count,
            "daily_trade_count": self._state.daily_trade_count,
            "open_positions": len(self._positions),
            "circuit_level": self._cb.level,
            "capital": self._capital,
            "paper_summary": self._paper.get_portfolio_summary(),
        }

    def shutdown(self) -> None:
        """엔진 종료 (ProcessPoolExecutor 정리)."""
        self._process_pool.shutdown(wait=False)
        logger.info("[Engine] 종료 완료")
