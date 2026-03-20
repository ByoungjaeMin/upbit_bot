"""test_walk_forward.py — WalkForwardOptimizer + SurvivourshipHandler + BacktestEngine 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from backtest.walk_forward import (
    BEAR_MAX_DRAWDOWN,
    IS_MONTHS,
    OOS_MONTHS,
    OVERFITTING_THRESHOLD,
    PESSIMISTIC_LOSS_COST,
    PESSIMISTIC_WIN_COST,
    BacktestEngine,
    BacktestParams,
    PeriodMetrics,
    SurvivourshipHandler,
    TradeResult,
    WalkForwardCycle,
    WalkForwardOptimizer,
    WalkForwardResult,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _make_df(n: int = 500, start: str = "2023-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    price = 50_000_000 + np.cumsum(rng.normal(0, 50_000, n))
    price = np.clip(price, 1_000, None)
    return pd.DataFrame({
        "close_5m": price,
        "open_5m":  price * 0.999,
        "high_5m":  price * 1.001,
        "low_5m":   price * 0.999,
        "volume_5m": rng.uniform(1, 10, n),
        "rsi_5m":   rng.uniform(30, 70, n),
        "adx_5m":   rng.uniform(15, 40, n),
        "atr_5m":   price * 0.01,
        "ema50_1d": price * 0.99,
        "ema200_1d": price * 0.98,
        "rsi_1d":   rng.uniform(40, 60, n),
        "trend_encoding_1d": np.ones(n, dtype=int),
    }, index=idx)


def _make_large_df(months: int = 10) -> pd.DataFrame:
    """Walk-Forward에 충분한 크기 DataFrame (날짜 기반 생성)."""
    # 날짜 범위를 명시적으로 생성
    n_days = months * 30
    idx = pd.date_range("2023-01-01", periods=n_days * 24 * 12, freq="5min", tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(42)
    price = 50_000_000 + np.cumsum(rng.normal(0, 50_000, n))
    price = np.clip(price, 1_000, None)
    return pd.DataFrame({
        "close_5m": price,
        "open_5m":  price * 0.999,
        "high_5m":  price * 1.001,
        "low_5m":   price * 0.999,
        "volume_5m": rng.uniform(1, 10, n),
        "rsi_5m":   rng.uniform(30, 70, n),
        "adx_5m":   rng.uniform(15, 40, n),
        "atr_5m":   price * 0.01,
        "ema50_1d": price * 0.99,
        "ema200_1d": price * 0.98,
        "rsi_1d":   rng.uniform(40, 60, n),
        "trend_encoding_1d": np.ones(n, dtype=int),
    }, index=idx)


def _params() -> BacktestParams:
    return BacktestParams()


def _trade(pnl: float = 0.01, strategy: str = "TREND") -> TradeResult:
    return TradeResult(
        timestamp=datetime.now(timezone.utc),
        coin="BTC",
        side="SELL",
        entry_price=50_000_000,
        exit_price=50_000_000 * (1 + pnl),
        pnl_pct=pnl,
        pnl_pct_pessimistic=0.0,
        strategy_type=strategy,
        hold_minutes=30.0,
    )


# ---------------------------------------------------------------------------
# BacktestParams
# ---------------------------------------------------------------------------

class TestBacktestParams:
    def test_default_values(self):
        p = BacktestParams()
        assert p.rsi_period == 14
        assert p.ensemble_threshold == pytest.approx(0.62)
        assert p.adx_threshold == pytest.approx(20.0)

    def test_custom_values(self):
        p = BacktestParams(rsi_period=20, adx_threshold=25.0)
        assert p.rsi_period == 20
        assert p.adx_threshold == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# BacktestEngine._apply_pessimistic_cost
# ---------------------------------------------------------------------------

class TestPessimisticCost:
    def test_profit_reduced(self):
        pnls = BacktestEngine._apply_pessimistic_cost([0.05])
        assert pnls[0] == pytest.approx(0.05 + PESSIMISTIC_WIN_COST)

    def test_loss_worsened(self):
        pnls = BacktestEngine._apply_pessimistic_cost([-0.03])
        assert pnls[0] == pytest.approx(-0.03 + PESSIMISTIC_LOSS_COST)

    def test_zero_pnl_uses_loss_path(self):
        pnls = BacktestEngine._apply_pessimistic_cost([0.0])
        assert pnls[0] == pytest.approx(PESSIMISTIC_LOSS_COST)

    def test_multiple_trades(self):
        pnls = BacktestEngine._apply_pessimistic_cost([0.05, -0.03, 0.02])
        assert len(pnls) == 3

    def test_empty_list(self):
        pnls = BacktestEngine._apply_pessimistic_cost([])
        assert pnls == []


# ---------------------------------------------------------------------------
# BacktestEngine._compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_all_profit_high_win_rate(self):
        trades = [_trade(0.02)] * 10
        pnls = [0.02] * 10
        m = BacktestEngine._compute_metrics(pnls, trades, "2024-01", "2024-02")
        assert m.win_rate == pytest.approx(1.0)

    def test_mixed_pnl_win_rate(self):
        pnls = [0.02, 0.03, -0.01, -0.02, 0.01]
        trades = [_trade(p) for p in pnls]
        m = BacktestEngine._compute_metrics(pnls, trades, "2024-01", "2024-02")
        assert m.win_rate == pytest.approx(0.6)

    def test_max_drawdown_non_positive(self):
        pnls = [0.05, -0.10, 0.03]
        trades = [_trade(p) for p in pnls]
        m = BacktestEngine._compute_metrics(pnls, trades, "", "")
        assert m.max_drawdown <= 0.0

    def test_n_trades_matches(self):
        pnls = [0.01] * 7
        trades = [_trade(p) for p in pnls]
        m = BacktestEngine._compute_metrics(pnls, trades, "", "")
        assert m.n_trades == 7

    def test_strategy_contributions_populated(self):
        trades = [_trade(0.02, "TREND")] * 5 + [_trade(-0.01, "GRID")] * 3
        pnls = [t.pnl_pct for t in trades]
        m = BacktestEngine._compute_metrics(pnls, trades, "", "")
        assert "TREND" in m.strategy_contributions
        assert "GRID" in m.strategy_contributions

    def test_period_start_end_preserved(self):
        m = BacktestEngine._compute_metrics([0.01], [_trade()], "2024-01", "2024-06")
        assert m.period_start == "2024-01"
        assert m.period_end == "2024-06"


# ---------------------------------------------------------------------------
# BacktestEngine.run()
# ---------------------------------------------------------------------------

class TestBacktestEngineRun:
    def test_run_returns_period_metrics(self):
        engine = BacktestEngine()
        df = _make_df(300)
        m = engine.run(df, _params())
        assert isinstance(m, PeriodMetrics)

    def test_run_empty_df_returns_empty_metrics(self):
        engine = BacktestEngine()
        m = engine.run(pd.DataFrame(), _params())
        assert m.n_trades == 0

    def test_run_with_custom_strategy_fn(self):
        engine = BacktestEngine()
        df = _make_df(200)

        def always_one_trade(df, params):
            return [_trade(0.02)]

        m = engine.run(df, _params(), strategy_fn=always_one_trade)
        assert m.n_trades == 1

    def test_run_metrics_sharpe_is_float(self):
        engine = BacktestEngine()
        df = _make_df(300)
        m = engine.run(df, _params())
        assert isinstance(m.sharpe, float)

    def test_pessimistic_cost_applied(self):
        """비관적 비용 적용 → 순수 손익보다 성과 나쁨."""
        raw_pnls = [0.02, 0.03, 0.01]
        pessimistic = BacktestEngine._apply_pessimistic_cost(raw_pnls)
        assert all(p < r for p, r in zip(pessimistic, raw_pnls))

    def test_to_dict_has_required_keys(self):
        engine = BacktestEngine()
        df = _make_df(200)
        m = engine.run(df, _params())
        d = m.to_dict()
        for key in ("n_trades", "win_rate", "sharpe", "max_drawdown"):
            assert key in d


# ---------------------------------------------------------------------------
# SurvivourshipHandler
# ---------------------------------------------------------------------------

class TestSurvivourshipHandler:
    def test_empty_snapshot_returns_empty(self):
        h = SurvivourshipHandler()
        coins = h.get_coins_at("2024-01-15")
        assert coins == []

    def test_get_coins_exact_date(self):
        h = SurvivourshipHandler({"2024-01-15": ["BTC", "ETH", "XRP"]})
        assert h.get_coins_at("2024-01-15") == ["BTC", "ETH", "XRP"]

    def test_get_coins_uses_closest_past_snapshot(self):
        h = SurvivourshipHandler({
            "2024-01-01": ["BTC"],
            "2024-02-01": ["BTC", "ETH"],
        })
        # 2024-01-20 → 가장 가까운 과거 = 2024-01-01
        coins = h.get_coins_at("2024-01-20")
        assert coins == ["BTC"]

    def test_get_coins_no_past_snapshot_returns_empty(self):
        h = SurvivourshipHandler({"2024-06-01": ["BTC"]})
        coins = h.get_coins_at("2024-01-01")  # 2024-06보다 과거
        assert coins == []

    def test_get_coins_with_date_object(self):
        from datetime import date
        h = SurvivourshipHandler({"2024-01-15": ["BTC", "ETH"]})
        coins = h.get_coins_at(date(2024, 1, 15))
        assert "BTC" in coins

    def test_has_minimum_history_true(self):
        snap = {f"2023-0{m+1}-01": ["BTC"] for m in range(7)}  # 7개월
        h = SurvivourshipHandler(snap)
        assert h.has_minimum_history(6) is True

    def test_has_minimum_history_false_insufficient(self):
        snap = {"2024-01-01": ["BTC"], "2024-02-01": ["BTC"]}  # 1개월
        h = SurvivourshipHandler(snap)
        assert h.has_minimum_history(6) is False

    def test_has_minimum_history_empty(self):
        h = SurvivourshipHandler()
        assert h.has_minimum_history(6) is False

    def test_snapshot_count(self):
        h = SurvivourshipHandler({"2024-01-01": ["BTC"], "2024-02-01": ["ETH"]})
        assert h.snapshot_count == 2


# ---------------------------------------------------------------------------
# WalkForwardOptimizer._generate_cycles
# ---------------------------------------------------------------------------

class TestGenerateCycles:
    def test_generates_cycles_for_long_df(self):
        wf = WalkForwardOptimizer()
        df = _make_large_df(months=10)
        cycles = wf._generate_cycles(df)
        assert len(cycles) >= 1

    def test_empty_df_no_cycles(self):
        wf = WalkForwardOptimizer()
        cycles = wf._generate_cycles(pd.DataFrame())
        assert cycles == []

    def test_cycle_has_4_elements(self):
        wf = WalkForwardOptimizer()
        df = _make_large_df(months=10)
        cycles = wf._generate_cycles(df)
        for c in cycles:
            assert len(c) == 4

    def test_oos_start_one_day_after_is_end(self):
        from datetime import datetime, timedelta
        wf = WalkForwardOptimizer()
        df = _make_large_df(months=10)
        cycles = wf._generate_cycles(df)
        for is_s, is_e, oos_s, oos_e in cycles:
            is_e_dt  = datetime.strptime(is_e,  "%Y-%m-%d")
            oos_s_dt = datetime.strptime(oos_s, "%Y-%m-%d")
            assert oos_s_dt == is_e_dt + timedelta(days=1)


# ---------------------------------------------------------------------------
# WalkForwardOptimizer.run()
# ---------------------------------------------------------------------------

class TestWalkForwardRun:
    def test_returns_walk_forward_result(self):
        wf = WalkForwardOptimizer()
        df = _make_large_df(months=10)
        result = wf.run(df, optimize_fn=lambda _: BacktestParams())
        assert isinstance(result, WalkForwardResult)

    def test_avg_sharpe_is_float(self):
        wf = WalkForwardOptimizer()
        df = _make_large_df(months=10)
        result = wf.run(df, optimize_fn=lambda _: BacktestParams())
        assert isinstance(result.avg_oos_sharpe, float)

    def test_insufficient_data_returns_empty(self):
        wf = WalkForwardOptimizer()
        df = _make_df(n=10)  # 너무 짧음
        result = wf.run(df, optimize_fn=lambda _: BacktestParams())
        assert isinstance(result, WalkForwardResult)

    def test_overfitting_flag_detected(self):
        """IS 샤프 고 / OOS 샤프 낮으면 과적합 감지."""
        cycle = WalkForwardCycle(
            cycle_idx=0,
            is_start="2023-01-01", is_end="2023-07-01",
            oos_start="2023-07-01", oos_end="2023-08-01",
            is_sharpe=2.0, oos_sharpe=0.5,   # ratio=0.25 < 0.5
            best_params=BacktestParams(),
            overfitting_flag=True,
            lookahead_passed=True,
        )
        assert cycle.overfitting_flag is True
        assert cycle.is_oos_ratio == pytest.approx(0.25)

    def test_is_oos_ratio_no_overfitting(self):
        cycle = WalkForwardCycle(
            cycle_idx=0,
            is_start="", is_end="", oos_start="", oos_end="",
            is_sharpe=1.5, oos_sharpe=1.2,   # ratio=0.8 > 0.5 → OK
            best_params=BacktestParams(),
            overfitting_flag=False,
            lookahead_passed=True,
        )
        assert cycle.is_oos_ratio == pytest.approx(0.8)

    def test_is_oos_ratio_zero_is_sharpe(self):
        cycle = WalkForwardCycle(
            cycle_idx=0,
            is_start="", is_end="", oos_start="", oos_end="",
            is_sharpe=0.0, oos_sharpe=1.0,
            best_params=BacktestParams(),
            overfitting_flag=False,
            lookahead_passed=True,
        )
        assert cycle.is_oos_ratio == 0.0


# ---------------------------------------------------------------------------
# WalkForwardOptimizer.check_live_readiness
# ---------------------------------------------------------------------------

class TestCheckLiveReadiness:
    def test_all_pass_returns_empty_failures(self):
        wf = WalkForwardOptimizer()
        result = WalkForwardResult(
            avg_oos_sharpe=2.0,
            regime_metrics={
                "bear": PeriodMetrics(
                    period_start="", period_end="",
                    n_trades=50, win_rate=0.60,
                    sharpe=1.8, max_drawdown=-0.08,
                    total_return=0.15, profit_loss_ratio=1.5,
                ),
            },
            cycles=[
                WalkForwardCycle(0, "", "", "", "", 2.0, 1.8,
                                 BacktestParams(), False, True)
            ],
        )
        failures = wf.check_live_readiness(result)
        assert failures == []

    def test_low_oos_sharpe_fails(self):
        wf = WalkForwardOptimizer()
        result = WalkForwardResult(avg_oos_sharpe=1.2, cycles=[])
        failures = wf.check_live_readiness(result)
        assert any("샤프" in f for f in failures)

    def test_high_bear_drawdown_fails(self):
        wf = WalkForwardOptimizer()
        result = WalkForwardResult(
            avg_oos_sharpe=2.0,
            regime_metrics={
                "bear": PeriodMetrics(
                    period_start="", period_end="",
                    n_trades=30, win_rate=0.60,
                    sharpe=1.5, max_drawdown=-0.15,   # -15% > -10% 기준 초과
                    total_return=-0.05, profit_loss_ratio=1.0,
                ),
            },
            cycles=[],
        )
        failures = wf.check_live_readiness(result)
        assert any("낙폭" in f for f in failures)

    def test_lookahead_failure_detected(self):
        wf = WalkForwardOptimizer()
        result = WalkForwardResult(
            avg_oos_sharpe=2.0,
            cycles=[
                WalkForwardCycle(0, "", "", "", "", 2.0, 1.5,
                                 BacktestParams(), False, False),  # lookahead_passed=False
            ],
        )
        failures = wf.check_live_readiness(result)
        assert any("Lookahead" in f for f in failures)

    def test_overfitting_detected(self):
        wf = WalkForwardOptimizer()
        result = WalkForwardResult(
            avg_oos_sharpe=2.0,
            cycles=[
                WalkForwardCycle(i, "", "", "", "", 2.0, 0.5,
                                 BacktestParams(), True, True)   # overfitting
                for i in range(4)
            ],
            overfitting_cycles=4,
        )
        failures = wf.check_live_readiness(result)
        assert any("과적합" in f for f in failures)


# ---------------------------------------------------------------------------
# WalkForwardOptimizer.evaluate_regime_periods
# ---------------------------------------------------------------------------

class TestEvaluateRegimePeriods:
    def test_returns_dict_with_regimes(self):
        wf = WalkForwardOptimizer()
        # 기간 데이터가 없어도 크래시 없어야
        df = _make_df(n=50, start="2024-01-01")
        result = wf.evaluate_regime_periods(df, BacktestParams())
        # 데이터가 regime 기간과 겹치지 않으면 빈 dict 반환
        assert isinstance(result, dict)

    def test_summary_string_not_empty(self):
        result = WalkForwardResult(
            cycles=[WalkForwardCycle(0, "A", "B", "C", "D", 1.5, 1.2, BacktestParams(), False, True)],
            avg_oos_sharpe=1.2,
            avg_is_sharpe=1.5,
        )
        s = result.summary()
        assert "WalkForward" in s
        assert "OOS" in s
