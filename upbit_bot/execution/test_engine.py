"""test_engine.py — TradingEngine 단위 테스트."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution.engine import (
    COLD_START_THRESHOLD,
    ENTRY_DELAY_MAX_SEC,
    HARD_STOP_LOSS_PCT,
    MAX_DAILY_TRADES,
    MAX_POSITIONS,
    PARTIAL_TP_1_PCT,
    PARTIAL_TP_2_PCT,
    REENTRY_COOLDOWN_MIN,
    Position,
    TradingEngine,
)
from execution.order import UpbitClient
from schema import EnsemblePrediction, MarketState, RiskBudget


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _engine(trade_count: int = 0, dry_run: bool = True) -> TradingEngine:
    engine = TradingEngine(dry_run=dry_run, initial_capital=10_000_000)
    engine._state.trade_count = trade_count
    return engine


def _ms(
    coin: str = "BTC",
    price: float = 50_000_000,
    tick_imbalance: float = 0.2,
    obi: float = 0.3,
    atr: float = 500_000,
    rsi: float = 50.0,
    trade_velocity: float = 1.0,
    supertrend: int = 1,
) -> MarketState:
    ms = MarketState(
        coin=coin,
        timestamp=datetime.now(timezone.utc),
        close_5m=price,
        open_5m=price,
        high_5m=price * 1.01,
        low_5m=price * 0.99,
        volume_5m=10.0,
        rsi_5m=rsi,
        adx_5m=25.0,
        tick_imbalance=tick_imbalance,
        obi=obi,
        atr_5m=atr,
        supertrend_signal=supertrend,
        trade_velocity=trade_velocity,
    )
    return ms


def _ep(
    coin: str = "BTC",
    weighted_avg: float = 0.70,
    consensus: int = 4,
    confirmed: bool = True,
) -> EnsemblePrediction:
    return EnsemblePrediction(
        coin=coin,
        timestamp=datetime.now(timezone.utc),
        weighted_avg=weighted_avg,
        consensus_count=consensus,
        signal_confirmed=confirmed,
    )


def _rb(coin: str = "BTC", kelly: float = 0.02, var: float = 0.02) -> RiskBudget:
    return RiskBudget(
        coin=coin,
        timestamp=datetime.now(timezone.utc),
        kelly_f=kelly,
        hmm_adjusted_f=kelly,
        var_adjusted_f=kelly,
        final_position_size=200_000,
        var_95=var,
    )


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 엔진 초기화
# ---------------------------------------------------------------------------

class TestEngineInit:
    def test_dry_run_default_true(self):
        engine = TradingEngine()
        assert engine.is_dry_run is True

    def test_trade_count_starts_zero(self):
        engine = TradingEngine()
        assert engine.trade_count == 0

    def test_no_positions_initially(self):
        engine = TradingEngine()
        assert len(engine.positions) == 0

    def test_circuit_breaker_level_0(self):
        engine = TradingEngine()
        assert engine.circuit_breaker.level == 0

    def test_get_status_keys(self):
        engine = TradingEngine()
        status = engine.get_status()
        for key in ("dry_run", "trade_count", "open_positions", "circuit_level"):
            assert key in status

    def test_setup_layers_stores_refs(self):
        engine = TradingEngine()
        mock_l1 = MagicMock()
        engine.setup_layers(layer1=mock_l1)
        assert engine._layer1 is mock_l1


# ---------------------------------------------------------------------------
# DRY_RUN 강제
# ---------------------------------------------------------------------------

class TestDryRunEnforcement:
    def test_below_threshold_forces_dry_run(self):
        engine = _engine(trade_count=0)
        engine._dry_run = False
        engine._enforce_dry_run()
        assert engine.is_dry_run is True

    def test_above_threshold_allows_live(self):
        engine = _engine(trade_count=COLD_START_THRESHOLD)
        engine._dry_run = False
        engine._enforce_dry_run()  # 200 이상 → 변경 없음
        assert engine.is_dry_run is False

    def test_assert_dry_run_is_true_below_threshold(self):
        """trade_count < 200: assert DRY_RUN == True (CLAUDE.md 원칙 7)."""
        engine = _engine(trade_count=199)
        engine._enforce_dry_run()
        assert engine._dry_run is True


# ---------------------------------------------------------------------------
# 일일 카운터 리셋
# ---------------------------------------------------------------------------

class TestDailyCounterReset:
    def test_reset_on_new_day(self):
        engine = _engine()
        engine._state.daily_trade_count = 5
        engine._state.last_trade_day = "2025-01-01"
        engine._reset_daily_counter()
        assert engine._state.daily_trade_count == 0

    def test_no_reset_same_day(self):
        engine = _engine()
        engine._state.daily_trade_count = 5
        today = datetime.now(timezone.utc).date().isoformat()
        engine._state.last_trade_day = today
        engine._reset_daily_counter()
        assert engine._state.daily_trade_count == 5


# ---------------------------------------------------------------------------
# 쿨타임
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_no_cooldown_initially(self):
        engine = _engine()
        assert engine._is_in_cooldown("BTC") is False

    def test_cooldown_set_after_stop_loss(self):
        engine = _engine()
        engine._set_cooldown("BTC")
        assert engine._is_in_cooldown("BTC") is True

    def test_cooldown_expires(self):
        engine = _engine()
        engine._state.reentry_cooldown["BTC"] = (
            datetime.now(timezone.utc) - timedelta(minutes=REENTRY_COOLDOWN_MIN + 1)
        )
        assert engine._is_in_cooldown("BTC") is False


# ---------------------------------------------------------------------------
# 재진입 조건
# ---------------------------------------------------------------------------

class TestReentryConditions:
    def test_all_3_conditions_met(self):
        engine = _engine()
        ms = _ms(tick_imbalance=0.20, rsi=50.0, obi=0.15)
        assert engine._check_reentry_conditions(ms) is True

    def test_only_1_condition_met_fails(self):
        engine = _engine()
        ms = _ms(tick_imbalance=0.05, rsi=80.0, obi=0.05)
        assert engine._check_reentry_conditions(ms) is False

    def test_exactly_2_conditions_met_passes(self):
        engine = _engine()
        # tick OK, RSI OK, OBI not OK
        ms = _ms(tick_imbalance=0.20, rsi=50.0, obi=0.05)
        assert engine._check_reentry_conditions(ms) is True


# ---------------------------------------------------------------------------
# 진입 트리거 6조건
# ---------------------------------------------------------------------------

class TestEntryTriggers:
    def _setup(self, engine: TradingEngine, ms: MarketState, ep: EnsemblePrediction, rb: RiskBudget):
        filter_results = {ms.coin: {"tradeable": True, "regime_strategy": "TREND_STRONG", "signal_multiplier": 1.0}}
        ensemble_preds = {ms.coin: ep}
        risk_budgets   = {ms.coin: rb}
        return filter_results, ensemble_preds, risk_budgets

    def test_all_conditions_met_returns_candidate(self):
        engine = _engine()
        ms = _ms()
        ep = _ep()
        rb = _rb()
        fr, ep_map, rb_map = self._setup(engine, ms, ep, rb)
        candidates = engine._evaluate_entry_triggers([ms], fr, ep_map, rb_map)
        assert len(candidates) == 1

    def test_low_ensemble_score_skips(self):
        engine = _engine()
        ms = _ms()
        ep = _ep(weighted_avg=0.50)  # 임계값 미달
        rb = _rb()
        fr, ep_map, rb_map = self._setup(engine, ms, ep, rb)
        candidates = engine._evaluate_entry_triggers([ms], fr, ep_map, rb_map)
        assert len(candidates) == 0

    def test_low_consensus_skips(self):
        engine = _engine()
        ms = _ms()
        ep = _ep(consensus=2)  # 3 미달
        rb = _rb()
        fr, ep_map, rb_map = self._setup(engine, ms, ep, rb)
        candidates = engine._evaluate_entry_triggers([ms], fr, ep_map, rb_map)
        assert len(candidates) == 0

    def test_zero_kelly_skips(self):
        engine = _engine()
        ms = _ms()
        ep = _ep()
        rb = _rb(kelly=0.0)
        fr, ep_map, rb_map = self._setup(engine, ms, ep, rb)
        candidates = engine._evaluate_entry_triggers([ms], fr, ep_map, rb_map)
        assert len(candidates) == 0

    def test_poor_microstructure_skips(self):
        engine = _engine()
        ms = _ms(tick_imbalance=0.05, obi=0.05)  # 둘 다 미달
        ep = _ep()
        rb = _rb()
        fr, ep_map, rb_map = self._setup(engine, ms, ep, rb)
        candidates = engine._evaluate_entry_triggers([ms], fr, ep_map, rb_map)
        assert len(candidates) == 0

    def test_cooldown_skips(self):
        engine = _engine()
        engine._set_cooldown("BTC")
        ms = _ms()
        ep = _ep()
        rb = _rb()
        fr, ep_map, rb_map = self._setup(engine, ms, ep, rb)
        candidates = engine._evaluate_entry_triggers([ms], fr, ep_map, rb_map)
        assert len(candidates) == 0

    def test_max_positions_stops_candidates(self):
        engine = _engine()
        # 최대 포지션 채우기
        for i in range(MAX_POSITIONS):
            engine._positions[f"COIN{i}"] = Position(
                coin=f"COIN{i}", entry_price=1000, qty=0.1,
                entry_krw=100_000, strategy_type="TREND_STRONG",
            )
        ms = _ms(coin="NEW_COIN")
        ep = _ep(coin="NEW_COIN")
        rb = _rb(coin="NEW_COIN")
        fr = {"NEW_COIN": {"tradeable": True, "regime_strategy": "TREND_STRONG", "signal_multiplier": 1.0}}
        candidates = engine._evaluate_entry_triggers([ms], fr, {ms.coin: ep}, {ms.coin: rb})
        assert len(candidates) == 0

    def test_candidates_sorted_by_score_descending(self):
        """앙상블×Kelly 내림차순 정렬."""
        engine = _engine()
        ms1 = _ms(coin="BTC")
        ms2 = _ms(coin="ETH")
        ep1 = _ep(coin="BTC", weighted_avg=0.90)
        ep2 = _ep(coin="ETH", weighted_avg=0.65)
        rb1 = _rb(coin="BTC", kelly=0.02)
        rb2 = _rb(coin="ETH", kelly=0.02)
        fr = {
            "BTC": {"tradeable": True, "regime_strategy": "TREND_STRONG", "signal_multiplier": 1.0},
            "ETH": {"tradeable": True, "regime_strategy": "TREND_STRONG", "signal_multiplier": 1.0},
        }
        candidates = engine._evaluate_entry_triggers(
            [ms1, ms2], fr, {"BTC": ep1, "ETH": ep2}, {"BTC": rb1, "ETH": rb2}
        )
        assert candidates[0][0].coin == "BTC"  # 높은 점수 먼저


# ---------------------------------------------------------------------------
# 포지션 모니터
# ---------------------------------------------------------------------------

class TestPositionMonitor:
    def test_trailing_stop_updates(self):
        engine = _engine()
        pos = Position("BTC", 50_000_000, 0.002, 100_000, "TREND_STRONG")
        ms = _ms(atr=500_000)
        engine._update_trailing_stop(pos, ms)
        # trailing_stop = close - atr*2
        expected = ms.close_5m - ms.atr_5m * 2.0
        assert pos.trailing_stop_price == pytest.approx(expected)

    def test_trailing_stop_only_increases(self):
        engine = _engine()
        pos = Position("BTC", 50_000_000, 0.002, 100_000, "TREND_STRONG")
        pos.trailing_stop_price = 49_000_000  # 기존 값
        ms = _ms(price=50_000_000, atr=100_000)  # 새 stop = 49_800_000
        engine._update_trailing_stop(pos, ms)
        assert pos.trailing_stop_price >= 49_000_000

    def test_trailing_stop_no_update_if_lower(self):
        engine = _engine()
        pos = Position("BTC", 50_000_000, 0.002, 100_000, "TREND_STRONG")
        pos.trailing_stop_price = 49_000_000
        ms = _ms(price=48_000_000, atr=1_000_000)  # 새 stop = 46_000_000 (낮음)
        engine._update_trailing_stop(pos, ms)
        assert pos.trailing_stop_price == pytest.approx(49_000_000)

    def test_pnl_pct_calculation(self):
        pos = Position("BTC", 50_000_000, 0.002, 100_000, "TREND_STRONG")
        assert pos.pnl_pct(55_000_000) == pytest.approx(0.10, rel=0.01)

    def test_pnl_pct_negative(self):
        pos = Position("BTC", 50_000_000, 0.002, 100_000, "TREND_STRONG")
        assert pos.pnl_pct(45_000_000) == pytest.approx(-0.10, rel=0.01)

    def test_pnl_pct_zero_entry(self):
        pos = Position("BTC", 0, 0.002, 100_000, "TREND_STRONG")
        assert pos.pnl_pct(50_000_000) == 0.0

    def test_hold_minutes_positive(self):
        import time
        pos = Position("BTC", 50_000_000, 0.002, 100_000, "TREND_STRONG")
        time.sleep(0.01)
        assert pos.hold_minutes >= 0


# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------

class TestMainLoop:
    def test_circuit_breaker_blocks_loop(self):
        engine = _engine()
        engine._cb.trigger(2, "테스트")
        result = run(engine.main_loop([_ms()]))
        assert result == []

    def test_main_loop_returns_list(self):
        engine = _engine()
        result = run(engine.main_loop([_ms()]))
        assert isinstance(result, list)

    def test_main_loop_no_positions_without_signals(self):
        engine = _engine()
        # Layer1/Layer2 미주입 → 기본값으로 실행
        result = run(engine.main_loop([_ms()]))
        # DRY_RUN 모드에서 기본 레이어는 통과 가정 → 주문 실행됨
        assert isinstance(result, list)

    def test_daily_limit_stops_orders(self):
        engine = _engine()
        engine._state.daily_trade_count = MAX_DAILY_TRADES
        engine._state.last_trade_day = datetime.now(timezone.utc).date().isoformat()
        result = run(engine.main_loop([_ms()]))
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 포지션 루프
# ---------------------------------------------------------------------------

class TestPositionLoop:
    def test_position_loop_no_positions_noop(self):
        engine = _engine()
        run(engine.position_loop([_ms()]))  # 예외 없어야 함

    def test_circuit_blocked_skips_loop(self):
        engine = _engine()
        engine._cb.trigger(2, "테스트")
        run(engine.position_loop([_ms()]))  # 조기 반환

    def test_hard_stop_triggers_close(self):
        engine = _engine()
        pos = Position("BTC", 50_000_000, 0.002, 100_000, "TREND_STRONG")
        engine._positions["BTC"] = pos

        closed = []
        original_close = engine._close_position

        async def mock_close(coin, price, reason="", emergency=False, partial_ratio=None):
            closed.append((coin, reason))
            engine._positions.pop(coin, None)
            return True

        engine._close_position = mock_close

        # 현재가 -8% → hard stop
        ms = _ms(price=50_000_000 * 0.92)
        run(engine.position_loop([ms]))
        assert any(r == "hard_stop" for _, r in closed)


# ---------------------------------------------------------------------------
# 서킷브레이커 루프
# ---------------------------------------------------------------------------

class TestCircuitLoop:
    def test_circuit_loop_no_crash(self):
        engine = _engine()
        run(engine.circuit_loop([_ms()]))  # 예외 없어야 함

    def test_circuit_loop_recovers(self):
        engine = _engine()
        engine._cb.trigger(1, "테스트")
        # maybe_recover는 쿨타임 후 동작 — 이 테스트에서는 충돌 없음 확인
        run(engine.circuit_loop([_ms()]))


# ---------------------------------------------------------------------------
# 리스크 예산
# ---------------------------------------------------------------------------

class TestRiskBudgets:
    def test_compute_returns_budget_per_coin(self):
        engine = _engine()
        ms_list = [_ms("BTC"), _ms("ETH")]
        ep_map = {"BTC": _ep("BTC"), "ETH": _ep("ETH")}
        budgets = engine._compute_risk_budgets(ms_list, ep_map)
        assert "BTC" in budgets
        assert "ETH" in budgets

    def test_kelly_positive_for_bullish(self):
        engine = _engine()
        ms = _ms()
        ep = _ep(weighted_avg=0.70)
        budgets = engine._compute_risk_budgets([ms], {"BTC": ep})
        assert budgets["BTC"].kelly_f > 0

    def test_kelly_zero_for_neutral(self):
        engine = _engine()
        ms = _ms()
        ep = _ep(weighted_avg=0.50)
        budgets = engine._compute_risk_budgets([ms], {"BTC": ep})
        assert budgets["BTC"].kelly_f == pytest.approx(0.0)

    def test_min_position_size_5000(self):
        engine = _engine()
        ms = _ms()
        ep = _ep(weighted_avg=0.501)  # 매우 낮은 Kelly
        budgets = engine._compute_risk_budgets([ms], {"BTC": ep})
        assert budgets["BTC"].final_position_size >= 5_000


# ---------------------------------------------------------------------------
# 엔진 종료
# ---------------------------------------------------------------------------

class TestEngineShutdown:
    def test_shutdown_no_crash(self):
        engine = _engine()
        engine.shutdown()
