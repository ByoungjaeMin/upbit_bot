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
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from risk.circuit_breaker import CircuitBreaker
from data.quality import DataQualityChecker
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
        yaml_config: dict | None = None,
    ) -> None:
        self._dry_run = dry_run
        self._capital = initial_capital

        # config.yaml 값으로 모듈 상수 오버라이드 (없으면 기본값 유지)
        _t = (yaml_config or {}).get("trading", {})
        self._max_positions       = int(_t.get("max_positions",       MAX_POSITIONS))
        self._min_hold_minutes    = float(_t.get("min_hold_minutes",  MIN_HOLD_MINUTES))
        self._reentry_cooldown    = float(_t.get("reentry_cooldown_min", REENTRY_COOLDOWN_MIN))
        self._max_daily_trades    = int(_t.get("max_daily_trades",    MAX_DAILY_TRADES))
        self._ensemble_threshold  = float(_t.get("ensemble_threshold", ENSEMBLE_THRESHOLD) if "ensemble_threshold" in _t else (yaml_config or {}).get("layer2", {}).get("threshold", ENSEMBLE_THRESHOLD))
        self._consensus_min       = int(_t.get("consensus_min", CONSENSUS_MIN) if "consensus_min" in _t else (yaml_config or {}).get("layer2", {}).get("consensus_min", CONSENSUS_MIN))
        self._var_max_pct         = float(_t.get("var_max_pct",       VAR_MAX_PCT))
        self._tick_imbalance_min  = float(_t.get("tick_imbalance_min", TICK_IMBALANCE_MIN))
        self._obi_min             = float(_t.get("obi_min",           OBI_MIN))
        self._reentry_tick_min    = float(_t.get("reentry_tick_min",  REENTRY_TICK_MIN))
        self._reentry_rsi_min     = float(_t.get("reentry_rsi_min",   REENTRY_RSI_MIN))
        self._reentry_rsi_max     = float(_t.get("reentry_rsi_max",   REENTRY_RSI_MAX))
        self._reentry_obi_min     = float(_t.get("reentry_obi_min",   REENTRY_OBI_MIN))
        self._hard_stop_loss_pct  = float(_t.get("hard_stop_loss_pct", HARD_STOP_LOSS_PCT))
        self._partial_tp1_pct     = float(_t.get("partial_tp1_pct",   PARTIAL_TP_1_PCT))
        self._partial_tp2_pct     = float(_t.get("partial_tp2_pct",   PARTIAL_TP_2_PCT))
        self._entry_delay_max_sec = float(_t.get("entry_delay_max_sec", ENTRY_DELAY_MAX_SEC))

        # 서킷브레이커 (최우선)
        self._cb = CircuitBreaker()

        # 데이터 품질 검증 (circuit_breaker 주입으로 Level 4 자동 발동)
        self._quality_checker = DataQualityChecker(circuit_breaker=self._cb)
        self._candle_builder: Any = None  # set_candle_builder()로 주입

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

        # Phase 자동 전환 감지 (DRY_RUN → 실거래 검토 알림)
        self._telegram_cb: Any = None          # main.py에서 set_telegram_callback()으로 주입
        self._cold_start_notified: bool = False  # 200건 달성 알림 1회 방지
        self._db_path: str | None = None       # main.py에서 set_db_path()로 주입

        # 가격 이력 (서킷브레이커 가격 급락 감지용, 코인당 최대 12개 = 1시간)
        self._price_history: dict[str, deque[tuple[datetime, float]]] = {}

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

    def set_candle_builder(self, candle_builder: Any) -> None:
        """CandleBuilder 주입 — DataQualityChecker 품질 검증용."""
        self._candle_builder = candle_builder

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

        # Step 1: 가격 급락 서킷브레이커 체크 — 레벨 에스컬레이션(1→2→3) 감지를 위해
        #         매수 차단 판단보다 반드시 먼저 실행해야 함.
        self._check_price_circuits(market_states)

        # Step 1.5: 매수 차단 확인 (Level 1+) — 신규 진입 판단이므로 is_buy_blocked() 사용
        if self._cb.is_buy_blocked():
            logger.warning("[Engine] 서킷브레이커 Level %d — 매수 차단 (포지션 모니터링은 유지)", self._cb.level)
            return decisions

        # Step 2: DRY_RUN 강제 체크
        self._enforce_dry_run()

        # Step 2.5: DataQualityChecker 7단계 품질 검증 — score < 0.5 코인 제외
        if self._candle_builder is not None:
            market_states = await self._filter_by_quality(market_states)

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
            if len(self._positions) >= self._max_positions:
                logger.info("[Engine] 최대 포지션 %d 초과 — %s 스킵", self._max_positions, ms.coin)
                break
            if self._state.daily_trade_count >= self._max_daily_trades:
                logger.info("[Engine] 일 거래 상한 %d 도달", self._max_daily_trades)
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
        # 청산 차단 확인 (Level 3+) — 포지션 청산은 SELL이므로 is_sell_blocked() 사용.
        # Level 2는 신규 진입만 차단; SELL은 허용하여 "전량 USDT 전환"이 가능하다.
        if self._cb.is_sell_blocked():
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

            # 하드 손절
            if pos.pnl_pct(current) <= self._hard_stop_loss_pct:
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
            if ep.weighted_avg < self._ensemble_threshold:
                logger.debug("[Engine] %s 앙상블 %.3f < %.2f", coin, ep.weighted_avg, self._ensemble_threshold)
                continue
            if ep.consensus_count < self._consensus_min:
                logger.debug("[Engine] %s consensus %d < %d", coin, ep.consensus_count, self._consensus_min)
                continue

            # ③ Kelly > 0
            if rb.var_adjusted_f <= 0:
                logger.debug("[Engine] %s Kelly ≤ 0 스킵", coin)
                continue

            # ④ VaR ≤ 자본 N%
            if rb.var_95 > self._var_max_pct:
                # 포지션 50% 축소 후 재판단
                rb.var_adjusted_f *= 0.5
                rb.final_position_size *= 0.5
                logger.info("[Engine] %s VaR %.3f > %.0f%% — 포지션 50%% 축소", coin, rb.var_95, self._var_max_pct * 100)
                if rb.var_adjusted_f <= 0:
                    continue

            # ⑤ 마이크로스트럭처 최소 확인
            if ms.tick_imbalance < self._tick_imbalance_min and ms.obi < self._obi_min:
                logger.debug("[Engine] %s 마이크로스트럭처 부족", coin)
                continue

            # ⑥ 동시 보유 포지션 < max
            if len(self._positions) >= self._max_positions:
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
            ms.tick_imbalance > self._reentry_tick_min,
            self._reentry_rsi_min <= ms.rsi_5m <= self._reentry_rsi_max,
            ms.obi > self._reentry_obi_min,
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
        if fr is None:
            # fr이 None이면 Layer1 결과 미존재 → HOLD
            return None
        # fr은 dict(캐시 직렬화 경로) 또는 FilterResult dataclass(정상 경로) 양쪽 모두 허용.
        # "HOLD" 폴백: regime_strategy 필드가 없는 경우 진입하지 않는 것이 안전함.
        strategy = fr.get("regime_strategy", "HOLD") if isinstance(fr, dict) else getattr(fr, "regime_strategy", "HOLD")
        if strategy == "HOLD":
            return None

        position_size = rb.final_position_size
        is_dry = self._dry_run

        # TREND: 0~90초 랜덤 지연 (front-running 방어)
        delay_sec = 0.0
        if strategy.startswith("TREND"):
            delay_sec = random.uniform(0, self._entry_delay_max_sec)
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
                trade_velocity=ms.trade_velocity if ms.trade_velocity is not None else 1.0,
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
            # trade_count가 COLD_START_THRESHOLD(200)에 도달 시 실거래 전환 검토 알림
            await self._check_cold_start_threshold()

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

        try:
            if emergency:
                # 강제 손절: SmartOrderRouter 우회 여부는 is_emergency 플래그로 처리
                # Level 2 이상 → split=False (전량 즉시)
                split = self._cb.level < 2
                status = await self._router.execute(req, split_emergency=split)
            else:
                # 일반 청산: 지정가 먼저, 10초 미체결 시 시장가
                status = await self._normal_close(req)
        except Exception as exc:
            # 청산 실패 시 서킷브레이커에 기록하고 포지션은 유지 (강제 삭제 금지)
            # position_loop가 다음 사이클에 재시도한다
            logger.error(
                "[Engine] 청산 실패 %s reason=%s: %s — 포지션 유지, 다음 사이클 재시도",
                coin, reason, exc,
            )
            self._cb.record_api_error()
            return False

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
            # 의도적 폴백: get_order 실패 시 미체결로 간주하고 시장가 전환.
            # 실제 체결됐더라도 cancel_with_race_guard → HTTP 400 "already done" →
            # PartialFillHandler가 FULLY_FILLED 처리하므로 이중 청산 없음.
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

        if pnl >= self._partial_tp2_pct and not pos.partial_exit_done.get("20pct"):
            pos.partial_exit_done["20pct"] = True
            await self._close_position(pos.coin, current_price, reason="partial_tp_20", partial_ratio=0.5)
            logger.info("[Engine] %s +%.0f%% 부분 익절 50%%", pos.coin, self._partial_tp2_pct * 100)

        elif pnl >= self._partial_tp1_pct and not pos.partial_exit_done.get("10pct"):
            pos.partial_exit_done["10pct"] = True
            await self._close_position(pos.coin, current_price, reason="partial_tp_10", partial_ratio=0.3)
            logger.info("[Engine] %s +%.0f%% 부분 익절 30%%", pos.coin, self._partial_tp1_pct * 100)

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
        """trade_count < 200 → DRY_RUN 강제.

        engine._dry_run 과 UpbitClient._dry_run 을 반드시 동시에 설정해야 한다.
        UpbitClient.place_limit_order / place_market_order 내부에서 _dry_run 플래그를
        직접 체크하므로, client 인스턴스도 동기화하지 않으면 실거래 API가 호출된다.
        """
        if self._state.trade_count < COLD_START_THRESHOLD:
            if not self._dry_run:
                logger.warning(
                    "[Engine] trade_count=%d < %d → DRY_RUN 강제",
                    self._state.trade_count, COLD_START_THRESHOLD,
                )
            self._dry_run = True
            # 실제 API 호출을 막는 것은 UpbitClient._dry_run 이므로 반드시 함께 설정
            self._client._dry_run = True
            assert self._dry_run is True  # CLAUDE.md 원칙 7번
            assert self._client._dry_run is True  # client 동기화 보장

    # ------------------------------------------------------------------
    # 텔레그램 / DB 콜백 주입 (외부 설정)
    # ------------------------------------------------------------------

    def set_telegram_callback(self, callback: Any) -> None:
        """텔레그램 전송 콜백 주입.

        main.py에서 TelegramBot.send를 주입:
            engine.set_telegram_callback(telegram_bot.send)

        미주입 시 200건 달성 알림이 로거로만 출력된다.
        """
        self._telegram_cb = callback

    def set_db_path(self, db_path: str) -> None:
        """SQLite DB 경로 주입.

        main.py에서 초기화 직후 호출:
            engine.set_db_path(str(cfg["db_path"]))

        미주입 시 7가지 체크리스트 조회를 건너뛴다.
        """
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Phase 자동 전환 감지
    # ------------------------------------------------------------------

    async def _check_cold_start_threshold(self) -> None:
        """trade_count가 COLD_START_THRESHOLD(200)에 도달하면 1회 텔레그램 알림.

        - 정확히 200 도달 시에만 1회 실행 (_cold_start_notified로 중복 방지)
        - DRY_RUN 강제 해제 여부는 _enforce_dry_run()이 담당
          이 메서드는 "검토 필요" 알림만 전송하며 실거래 전환을 자동으로 하지 않는다
        - 운영자가 직접 확인 후 수동으로 DRY_RUN=false 설정
        """
        if self._cold_start_notified:
            return
        if self._state.trade_count != COLD_START_THRESHOLD:
            return

        self._cold_start_notified = True
        logger.info(
            "[Engine] trade_count=%d 달성 — 실거래 전환 검토 알림 전송",
            self._state.trade_count,
        )

        # 7가지 체크리스트 조회 (IO 블로킹이므로 to_thread)
        failures = await asyncio.to_thread(self._run_live_readiness_check)

        if failures:
            status_line = f"⚠️ 미통과 항목 {len(failures)}개 — 실거래 전환 보류"
        else:
            status_line = "✅ 7가지 기준 모두 통과 — 실거래 전환 가능"

        msg_lines = [
            "<b>🎯 페이퍼 트레이딩 200건 달성</b>",
            "실거래 전환 검토를 시작하세요.",
            "",
            f"<b>7가지 체크리스트 현황:</b> {status_line}",
        ]
        for f in failures:
            msg_lines.append(f"  • {f}")
        msg_lines += [
            "",
            "⚠️ DRY_RUN=false 설정 전 모든 항목을 직접 확인하세요.",
            "⚠️ trade_count가 200 미만으로 리셋되면 DRY_RUN이 자동 재강제됩니다.",
        ]
        msg = "\n".join(msg_lines)

        if self._telegram_cb is not None:
            try:
                await self._telegram_cb(msg, priority=1)
            except Exception as exc:
                logger.error("[Engine] 200건 달성 텔레그램 알림 실패: %s", exc)
        else:
            logger.warning("[Engine] telegram_cb 미주입 — 200건 알림 로거 출력:\n%s", msg)

    def _run_live_readiness_check(self) -> list[str]:
        """실전 전환 7가지 체크리스트 조회 (동기 — asyncio.to_thread로 호출).

        항목:
          ① 샤프비율 > 1.5         (wf_summary)
          ② 최대낙폭 < 20%         (trades)
          ③ 승률 > 55%             (trades)
          ④ 하락장 낙폭 < 10%      (wf_summary.failures_json)
          ⑤ Lookahead Bias 0개     (wf_summary.failures_json)
          ⑥ Monte Carlo p < 0.05  (backtest_results)
          ⑦ DRY_RUN 48시간 정상   (trades 최초 기록 기준)

        Returns:
            미통과 항목 설명 목록 (비어있으면 실전 전환 가능)
        """
        import json as _json
        import sqlite3 as _sqlite3

        failures: list[str] = []

        if not self._db_path:
            raise RuntimeError(
                "_run_live_readiness_check: db_path 미설정 — "
                "set_db_path() 호출 필요"
            )

        try:
            conn = _sqlite3.connect(str(self._db_path))
            conn.row_factory = _sqlite3.Row
            try:
                # ─── ①④⑤: Walk-Forward 결과 ─────────────────────────
                try:
                    wf_row = conn.execute(
                        "SELECT avg_oos_sharpe, failures_json"
                        " FROM wf_summary ORDER BY run_at DESC LIMIT 1"
                    ).fetchone()
                except _sqlite3.OperationalError:
                    wf_row = None  # 테이블 미생성

                if wf_row is None:
                    failures.append("① Walk-Forward 결과 없음 (run_walk_forward.py 실행 필요)")
                    failures.append("④ 하락장 낙폭 — Walk-Forward 미실행")
                    failures.append("⑤ Lookahead Bias — Walk-Forward 미실행")
                else:
                    sharpe = float(wf_row["avg_oos_sharpe"] or 0.0)
                    if sharpe < 1.5:
                        failures.append(f"① OOS 샤프비율 {sharpe:.3f} < 1.5")

                    wf_fails: list[str] = _json.loads(wf_row["failures_json"] or "[]")
                    for f in wf_fails:
                        if "하락장" in f:
                            failures.append(f"④ {f}")
                        elif "Lookahead" in f:
                            failures.append(f"⑤ {f}")
                        # 과적합 경고는 7가지 기준에 없으므로 생략

                # ─── ②③: trades 테이블 직접 계산 ────────────────────
                try:
                    trade_rows = conn.execute(
                        "SELECT pnl_pct FROM trades"
                        " WHERE side='SELL' AND pnl_pct IS NOT NULL ORDER BY timestamp"
                    ).fetchall()
                except _sqlite3.OperationalError:
                    trade_rows = []

                if len(trade_rows) < 30:
                    failures.append(
                        f"② 최대낙폭 — SELL 체결 {len(trade_rows)}건 < 30건 (데이터 부족)"
                    )
                    failures.append(
                        f"③ 승률 — SELL 체결 {len(trade_rows)}건 < 30건 (데이터 부족)"
                    )
                else:
                    import numpy as _np
                    pnls = _np.array([float(r["pnl_pct"]) for r in trade_rows])

                    # ② 최대낙폭
                    cum = _np.cumprod(1.0 + pnls)
                    roll_max = _np.maximum.accumulate(cum)
                    mdd = float(_np.min((cum - roll_max) / roll_max))
                    if mdd < -0.20:
                        failures.append(f"② 최대낙폭 {mdd:.1%} < -20%")

                    # ③ 승률
                    wr = float(_np.mean(pnls > 0))
                    if wr < 0.55:
                        failures.append(f"③ 승률 {wr:.1%} < 55%")

                # ─── ⑥: Monte Carlo ────────────────────────────────
                try:
                    mc_row = conn.execute(
                        "SELECT p_value, passed FROM backtest_results"
                        " WHERE type='monte_carlo' ORDER BY run_at DESC LIMIT 1"
                    ).fetchone()
                except _sqlite3.OperationalError:
                    mc_row = None

                if mc_row is None:
                    failures.append("⑥ Monte Carlo 결과 없음 (주간 백테스트 실행 필요)")
                elif not mc_row["passed"]:
                    failures.append(f"⑥ Monte Carlo p-value={mc_row['p_value']:.4f} >= 0.05")

                # ─── ⑦: DRY_RUN 48시간 ─────────────────────────────
                try:
                    ts_row = conn.execute(
                        "SELECT MIN(timestamp) as first_ts FROM trades WHERE is_dry_run=1"
                    ).fetchone()
                except _sqlite3.OperationalError:
                    ts_row = None

                if ts_row and ts_row["first_ts"]:
                    try:
                        ts_str: str = ts_row["first_ts"]
                        # ISO 형식 정규화 (타임존 없으면 UTC 가정)
                        if "+" not in ts_str and not ts_str.endswith("Z"):
                            ts_str += "+00:00"
                        first_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        elapsed_h = (
                            datetime.now(timezone.utc) - first_ts
                        ).total_seconds() / 3600
                        if elapsed_h < 48:
                            failures.append(
                                f"⑦ DRY_RUN 운영 {elapsed_h:.1f}시간 < 48시간"
                            )
                    except (ValueError, TypeError) as exc:
                        failures.append(f"⑦ DRY_RUN 시작 시각 파싱 불가: {exc}")
                else:
                    failures.append("⑦ DRY_RUN 거래 기록 없음")

            finally:
                conn.close()

        except Exception as exc:
            logger.error("[Engine] 체크리스트 조회 오류: %s", exc)
            raise

        return failures

    def _reset_daily_counter(self) -> None:
        """날짜 변경 시 일일 거래 횟수 리셋."""
        today = datetime.now(timezone.utc).date().isoformat()
        if self._state.last_trade_day != today:
            self._state.daily_trade_count = 0
            self._state.last_trade_day = today

    async def _filter_by_quality(
        self, market_states: list[MarketState]
    ) -> list[MarketState]:
        """DataQualityChecker score < 0.5 코인을 메인 루프에서 제외한다.

        CandleBuilder가 미주입(None)이면 호출 자체를 하지 않으므로(Step 2.5 가드)
        여기서는 항상 self._candle_builder가 유효하다고 가정한다.
        validate_pipeline이 async이므로 await 처리한다.
        """
        passed: list[MarketState] = []
        for ms in market_states:
            df = self._candle_builder._build_df_with_indicators(ms.coin, "5m")
            if df is None or df.empty:
                # DataFrame 없으면 품질 검증 불가 — 통과 처리 (데이터 부족은 Layer1이 처리)
                passed.append(ms)
                continue
            ws_price = ms.close_5m if ms.close_5m > 0 else None
            _, score, _ = await self._quality_checker.validate_pipeline(
                df, "5m", ms.coin, ws_price=ws_price
            )
            if score < 0.5:
                logger.warning(
                    "[Engine] %s 데이터 품질 score=%.3f < 0.5 — 해당 코인 루프 스킵",
                    ms.coin, score,
                )
                continue
            passed.append(ms)
        return passed

    def _check_price_circuits(self, market_states: list[MarketState]) -> None:
        """각 코인의 현재가를 이력에 기록하고 가격 급락 서킷브레이커를 체크한다.

        5분 루프 기준으로 이력을 관리한다:
          - price_1m_ago:  1~6분 전 (직전 5분봉 close) 근사값
          - price_10m_ago: 9분 이상 전 (2개 이전 5분봉 이상) 근사값

        가격 이력이 충분하지 않은 초기 구동 시에는 체크를 건너뛴다.
        """
        now = datetime.now(timezone.utc)
        for ms in market_states:
            coin = ms.coin
            current = ms.close_5m
            if current <= 0:
                continue

            if coin not in self._price_history:
                self._price_history[coin] = deque(maxlen=12)
            history = self._price_history[coin]
            history.append((now, current))

            # 1분 전 / 10분 전 가격 근사 (이력 부족 시 current로 대체 → 급락 미발동)
            price_1m_ago = current
            price_10m_ago = current
            for ts, price in history:
                age_min = (now - ts).total_seconds() / 60
                if 1 <= age_min <= 6 and price_1m_ago == current:
                    price_1m_ago = price
                if age_min >= 9 and price_10m_ago == current:
                    price_10m_ago = price

            self._cb.check_price_drop(
                coin=coin,
                price_1m_ago=price_1m_ago,
                price_10m_ago=price_10m_ago,
                current_price=current,
                daily_loss_pct=self._state.daily_loss_pct,
                cumulative_loss_24h=self._state.daily_loss_pct,
            )

    def _is_in_cooldown(self, coin: str) -> bool:
        """재진입 쿨타임 확인."""
        expire = self._state.reentry_cooldown.get(coin)
        if expire is None:
            return False
        return datetime.now(timezone.utc) < expire

    def _set_cooldown(self, coin: str) -> None:
        """재진입 쿨타임 설정."""
        self._state.reentry_cooldown[coin] = (
            datetime.now(timezone.utc) + timedelta(minutes=self._reentry_cooldown)
        )

    # ------------------------------------------------------------------
    # 병렬 실행 헬퍼
    # ------------------------------------------------------------------

    async def _run_layer1_parallel(
        self, market_states: list[MarketState]
    ) -> dict[str, Any]:
        """Layer 1 필터 병렬 실행 (asyncio.gather)."""
        if self._layer1 is None:
            # 테스트 전용 폴백: Layer1 미주입 시 전 코인을 tradeable=True로 처리.
            # 실거래 기동 시에는 setup_layers()로 반드시 Layer1을 주입해야 한다.
            logger.error(
                "[Engine] Layer1 미주입 — 전 코인(%d개) 필터 없이 통과 처리. "
                "실거래라면 setup_layers(layer1=...) 호출 필요.",
                len(market_states),
            )
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
            # 테스트 전용 폴백: Layer2 미주입 시 임의 앙상블 점수(0.65)로 신호 확정.
            # 실거래에서 이 경로가 실행되면 ML 검증 없이 매수 신호가 통과된다.
            # 실거래 기동 시에는 setup_layers(layer2=...) 호출 필요.
            logger.error(
                "[Engine] Layer2 미주입 — ML 앙상블 없이 신호 확정(signal_confirmed=True) 처리. "
                "실거래라면 setup_layers(layer2=...) 호출 필요.",
            )
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
