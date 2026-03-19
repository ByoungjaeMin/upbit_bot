"""test_trend.py — TrendStrategy 단위 테스트."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_BOT_DIR = Path(__file__).parent.parent
if str(_BOT_DIR) not in sys.path:
    sys.path.insert(0, str(_BOT_DIR))

from schema import FilterResult, EnsemblePrediction, MarketState
from strategies.trend import TrendStrategy


def _make_filter(regime: str = "TREND_STRONG", tradeable: bool = True) -> FilterResult:
    return FilterResult(
        coin="KRW-BTC",
        timestamp=datetime.now(timezone.utc),
        tradeable=tradeable,
        regime_strategy=regime,
        signal_multiplier=1.0,
        adx_value=35.0 if regime == "TREND_STRONG" else 25.0,
        supertrend_direction=1,
        atr_value=500_000,
    )


def _make_ensemble(confirmed: bool = True, prob: float = 0.70) -> EnsemblePrediction:
    return EnsemblePrediction(
        coin="KRW-BTC",
        timestamp=datetime.now(timezone.utc),
        per_model_probs={"xgb": prob, "lgb": prob},
        weighted_avg=prob,
        consensus_count=3,
        signal_confirmed=confirmed,
    )


def _make_market(rsi: float = 55.0, supertrend: int = 1, adx: float = 35.0) -> MarketState:
    ms = MarketState(
        coin="KRW-BTC",
        timestamp=datetime.now(timezone.utc),
    )
    ms.rsi_5m = rsi
    ms.supertrend_signal = supertrend
    ms.adx_5m = adx
    ms.close_5m = 50_000_000
    ms.ema99_5m = 48_000_000
    return ms


class TestTrendStrategyEntry:
    def setup_method(self):
        self.strategy = TrendStrategy()

    def test_enter_signal_strong(self):
        sig = self.strategy.evaluate_entry(
            _make_filter("TREND_STRONG"),
            _make_ensemble(),
            _make_market(adx=35.0),
        )
        assert sig is not None
        assert sig.action == "ENTER"
        assert sig.strategy_type == "TREND_STRONG"

    def test_enter_signal_normal(self):
        sig = self.strategy.evaluate_entry(
            _make_filter("TREND_NORMAL"),
            _make_ensemble(),
            _make_market(adx=25.0),
        )
        assert sig is not None
        assert sig.strategy_type == "TREND_NORMAL"

    def test_no_entry_when_not_tradeable(self):
        sig = self.strategy.evaluate_entry(
            _make_filter(tradeable=False),
            _make_ensemble(),
            _make_market(),
        )
        assert sig is None

    def test_no_entry_when_signal_not_confirmed(self):
        sig = self.strategy.evaluate_entry(
            _make_filter(),
            _make_ensemble(confirmed=False),
            _make_market(),
        )
        assert sig is None

    def test_no_entry_when_rsi_overbought(self):
        sig = self.strategy.evaluate_entry(
            _make_filter(),
            _make_ensemble(),
            _make_market(rsi=80.0),
        )
        assert sig is None

    def test_no_entry_when_rsi_oversold(self):
        sig = self.strategy.evaluate_entry(
            _make_filter(),
            _make_ensemble(),
            _make_market(rsi=30.0),
        )
        assert sig is None

    def test_no_entry_when_supertrend_down(self):
        sig = self.strategy.evaluate_entry(
            _make_filter(),
            _make_ensemble(),
            _make_market(supertrend=-1),
        )
        assert sig is None

    def test_no_entry_when_grid_regime(self):
        sig = self.strategy.evaluate_entry(
            _make_filter("GRID"),
            _make_ensemble(),
            _make_market(),
        )
        assert sig is None

    def test_no_entry_when_adx_too_low_for_strong(self):
        sig = self.strategy.evaluate_entry(
            _make_filter("TREND_STRONG"),
            _make_ensemble(),
            _make_market(adx=28.0),  # < 30
        )
        assert sig is None


class TestTrendStrategyExit:
    def setup_method(self):
        self.strategy = TrendStrategy()
        self.entry_time = datetime.now(timezone.utc) - timedelta(hours=1)

    def test_exit_on_trailing_stop(self):
        sig = self.strategy.evaluate_exit(
            "KRW-BTC",
            _make_market(),
            entry_price=50_000_000,
            entry_time=self.entry_time,
            trailing_stop_triggered=True,
        )
        assert sig is not None
        assert sig.action == "EXIT"

    def test_exit_on_supertrend_reversal(self):
        sig = self.strategy.evaluate_exit(
            "KRW-BTC",
            _make_market(supertrend=-1),
            entry_price=50_000_000,
            entry_time=self.entry_time,
        )
        assert sig is not None
        assert sig.action == "EXIT"

    def test_exit_on_rsi_below_threshold(self):
        sig = self.strategy.evaluate_exit(
            "KRW-BTC",
            _make_market(rsi=35.0),
            entry_price=50_000_000,
            entry_time=self.entry_time,
        )
        assert sig is not None
        assert sig.action == "EXIT"

    def test_partial_exit_on_5pct_gain(self):
        ms = _make_market()
        ms.close_5m = 52_600_000  # +5.2%
        sig = self.strategy.evaluate_exit(
            "KRW-BTC",
            ms,
            entry_price=50_000_000,
            entry_time=self.entry_time,
        )
        assert sig is not None
        assert sig.action == "PARTIAL_EXIT"

    def test_no_exit_when_all_good(self):
        sig = self.strategy.evaluate_exit(
            "KRW-BTC",
            _make_market(rsi=55.0, supertrend=1),
            entry_price=50_000_000,
            entry_time=self.entry_time,
        )
        assert sig is None

    def test_exit_on_max_hold_time(self):
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        sig = self.strategy.evaluate_exit(
            "KRW-BTC",
            _make_market(rsi=55.0, supertrend=1),
            entry_price=50_000_000,
            entry_time=old_time,
            strategy_type="TREND_STRONG",
        )
        assert sig is not None
        assert sig.action == "EXIT"
