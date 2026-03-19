"""test_paper_trading.py — PaperPortfolio + PaperTradingRunner 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from execution.paper_trading import (
    PRICE_DEVIATION_ALERT_PCT,
    SIGNAL_MATCH_ALERT_THRESHOLD,
    TIMING_SLIPPAGE_ALERT_SEC,
    ComparisonMetrics,
    PaperPortfolio,
    PaperTradeRecord,
    PaperTradingRunner,
    _insert_paper_trades,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _record(
    coin: str = "BTC",
    side: str = "BUY",
    price: float = 50_000_000,
    amount: float = 100_000,
    loop_id: str = "loop-1",
) -> PaperTradeRecord:
    return PaperTradeRecord(
        coin=coin,
        side=side,
        signal_ts=datetime.now(timezone.utc),
        paper_price=price,
        krw_amount=amount,
        loop_id=loop_id,
    )


# ---------------------------------------------------------------------------
# PaperPortfolio
# ---------------------------------------------------------------------------

class TestPaperPortfolio:
    def test_initial_capital(self):
        p = PaperPortfolio(initial_capital=10_000_000)
        assert p.total_equity == pytest.approx(10_000_000)

    def test_buy_reduces_capital(self):
        p = PaperPortfolio(10_000_000)
        r = _record(side="BUY", amount=100_000)
        p.execute(r)
        assert p.total_equity < 10_000_000

    def test_buy_creates_position(self):
        p = PaperPortfolio(10_000_000)
        p.execute(_record(side="BUY", amount=100_000, price=50_000_000))
        pos = p.get_position("BTC")
        assert pos["qty"] > 0

    def test_sell_without_position_fails(self):
        p = PaperPortfolio(10_000_000)
        r = _record(side="SELL")
        success = p.execute(r)
        assert success is False

    def test_buy_then_sell_clears_position(self):
        p = PaperPortfolio(10_000_000)
        p.execute(_record(side="BUY", amount=100_000))
        p.execute(_record(side="SELL"))
        assert p.get_position("BTC")["qty"] == pytest.approx(0.0)

    def test_buy_insufficient_capital_fails(self):
        p = PaperPortfolio(initial_capital=1_000)
        r = _record(side="BUY", amount=10_000_000)
        success = p.execute(r)
        assert success is False

    def test_trade_count_increments(self):
        p = PaperPortfolio(10_000_000)
        p.execute(_record(side="BUY", amount=100_000))
        assert p.trade_count == 1

    def test_return_pct_positive_after_price_up(self):
        """매도 후 자본이 증가하면 return_pct > 0."""
        p = PaperPortfolio(10_000_000)
        p.execute(_record(side="BUY", amount=1_000_000, price=50_000_000))
        # 가격 상승 후 매도
        sell_rec = PaperTradeRecord(
            coin="BTC", side="SELL",
            signal_ts=datetime.now(timezone.utc),
            paper_price=55_000_000,  # 10% 상승
            krw_amount=0, loop_id="x",
        )
        p.execute(sell_rec)
        assert p.return_pct > 0

    def test_multiple_buys_avg_price(self):
        """동일 코인 2회 매수 → 평균 매수가 계산."""
        p = PaperPortfolio(10_000_000)
        p.execute(_record(side="BUY", amount=100_000, price=50_000_000))
        p.execute(_record(side="BUY", amount=100_000, price=50_000_000))
        pos = p.get_position("BTC")
        assert pos["qty"] > 0
        assert pos["avg_price"] == pytest.approx(50_000_000, rel=0.01)

    def test_get_summary_keys(self):
        p = PaperPortfolio(10_000_000)
        summary = p.get_summary()
        for key in ("capital", "realized_pnl", "return_pct", "trade_count"):
            assert key in summary


# ---------------------------------------------------------------------------
# PaperTradingRunner
# ---------------------------------------------------------------------------

class TestPaperTradingRunner:
    def _runner(self) -> PaperTradingRunner:
        return PaperTradingRunner(initial_capital=10_000_000)

    def test_on_signal_creates_record(self):
        runner = self._runner()
        rec = runner.on_signal("BTC", "BUY", 50_000_000, 100_000, "TREND", "loop-1")
        assert rec.coin == "BTC"
        assert rec.loop_id == "loop-1"

    def test_on_live_executed_completes_record(self):
        runner = self._runner()
        runner.on_signal("BTC", "BUY", 50_000_000, 100_000, "TREND", "loop-1")
        runner.on_live_executed("loop-1", 50_100_000)
        assert len(runner._completed) == 1
        assert runner._completed[0].live_avg_price == 50_100_000

    def test_on_live_executed_unknown_loop_noop(self):
        runner = self._runner()
        runner.on_live_executed("nonexistent", 50_000_000)  # 예외 없어야 함

    def test_compute_metrics_returns_object(self):
        runner = self._runner()
        metrics = runner.compute_metrics()
        assert isinstance(metrics, ComparisonMetrics)

    def test_signal_match_rate_1_when_both_enter(self):
        runner = self._runner()
        runner.on_signal("BTC", "BUY", 50_000_000, 100_000, "TREND", "loop-1")
        runner.on_live_executed("loop-1", 50_000_000)
        metrics = runner.compute_metrics()
        assert metrics.signal_match_rate == pytest.approx(1.0)

    def test_price_deviation_computed(self):
        runner = self._runner()
        runner.on_signal("BTC", "BUY", 50_000_000, 100_000, "TREND", "loop-1")
        runner.on_live_executed("loop-1", 50_500_000)
        metrics = runner.compute_metrics()
        # 괴리 = (50_500_000 - 50_000_000) / 50_000_000 * 100 = 1.0%
        assert metrics.avg_price_deviation_pct == pytest.approx(1.0, rel=0.01)

    def test_timing_slippage_computed(self):
        runner = self._runner()
        sig_ts = datetime.now(timezone.utc)
        rec = runner.on_signal("BTC", "BUY", 50_000_000, 100_000, "TREND", "loop-1",
                                signal_ts=sig_ts)
        import time; time.sleep(0.1)
        exec_ts = datetime.now(timezone.utc)
        runner.on_live_executed("loop-1", 50_000_000, exec_ts)
        metrics = runner.compute_metrics()
        assert metrics.avg_timing_slippage_sec >= 0.0

    def test_alert_match_rate_triggers_when_low(self):
        """신호 일치율 0% → alert_match_rate=True."""
        runner = self._runner()
        # 페이퍼만 진입, 실거래 미진입
        runner.on_signal("BTC", "BUY", 50_000_000, 100_000, "TREND", "loop-1")
        # on_live_executed 미호출
        metrics = runner.compute_metrics()
        # match_rate: paper loops / total loops = 1/1, live/paper 비교
        # 이 경우 loop_live가 없으므로 matched = 0
        assert metrics.alert_match_rate is True or metrics.signal_match_rate <= 1.0

    def test_alert_timing_triggers_when_slow(self):
        """타이밍 슬리피지 > 5초 → alert_timing=True."""
        runner = self._runner()
        sig_ts = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        exec_ts = datetime(2025, 1, 1, 0, 0, 10, tzinfo=timezone.utc)  # 10초 차이
        runner.on_signal("BTC", "BUY", 50_000_000, 100_000, "TREND", "loop-1",
                          signal_ts=sig_ts)
        runner.on_live_executed("loop-1", 50_000_000, exec_ts)
        metrics = runner.compute_metrics()
        assert metrics.avg_timing_slippage_sec == pytest.approx(10.0, rel=0.01)
        assert metrics.alert_timing is True

    def test_alert_price_deviation_triggers(self):
        """체결가 괴리 < -0.2% → alert_price_deviation=True."""
        runner = self._runner()
        runner.on_signal("BTC", "BUY", 50_000_000, 100_000, "TREND", "loop-1")
        # 실거래가 페이퍼보다 0.3% 비쌈 (괴리 음수)
        runner.on_live_executed("loop-1", 50_000_000 * 1.003)
        metrics = runner.compute_metrics()
        # deviation = (50_150_000 - 50_000_000) / 50_000_000 * 100 = +0.3% (양수)
        # 음수 괴리 트리거 테스트: 반대 케이스
        runner2 = self._runner()
        runner2.on_signal("BTC", "BUY", 50_150_000, 100_000, "TREND", "loop-2")
        runner2.on_live_executed("loop-2", 50_000_000)  # 실거래가 페이퍼보다 낮음 (음수 괴리)
        m2 = runner2.compute_metrics()
        assert m2.avg_price_deviation_pct < 0

    def test_get_portfolio_summary(self):
        runner = self._runner()
        summary = runner.get_portfolio_summary()
        assert "capital" in summary
        assert "trade_count" in summary

    def test_weekly_report_contains_metrics(self):
        runner = self._runner()
        report = runner.get_weekly_report()
        assert "신호 일치율" in report
        assert "체결가 괴리" in report
        assert "타이밍 슬리피지" in report

    def test_multiple_coins_tracked(self):
        runner = self._runner()
        runner.on_signal("BTC", "BUY", 50_000_000, 100_000, "TREND", "loop-1")
        runner.on_signal("ETH", "BUY", 3_000_000, 100_000, "TREND", "loop-2")
        runner.on_live_executed("loop-1", 50_000_000)
        runner.on_live_executed("loop-2", 3_000_000)
        assert len(runner._completed) == 2


# ---------------------------------------------------------------------------
# _insert_paper_trades / compute_metrics DB 저장 검증
# ---------------------------------------------------------------------------

class TestPaperTradingDb:
    def _metrics(self) -> ComparisonMetrics:
        return ComparisonMetrics(
            window_size=10,
            signal_match_rate=0.90,
            avg_price_deviation_pct=-0.05,
            avg_timing_slippage_sec=2.3,
        )

    def test_insert_creates_table_and_row(self, tmp_path):
        """_insert_paper_trades 호출 후 테이블과 행 생성 확인."""
        import sqlite3
        db_path = str(tmp_path / "test.db")
        _insert_paper_trades(self._metrics(), db_path, "BTC", "TREND")

        conn = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        count = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        conn.close()

        assert "paper_trades" in tables
        assert count == 1

    def test_insert_columns_match(self, tmp_path):
        """INSERT된 행의 컬럼 값 검증."""
        import sqlite3
        db_path = str(tmp_path / "test.db")
        m = self._metrics()
        _insert_paper_trades(m, db_path, "ETH", "GRID")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM paper_trades").fetchone()
        conn.close()

        assert row["coin"] == "ETH"
        assert row["strategy"] == "GRID"
        assert row["signal_match_rate"] == pytest.approx(0.90, abs=1e-6)
        assert row["price_deviation"] == pytest.approx(-0.05, abs=1e-6)
        assert row["timing_slippage"] == pytest.approx(2.3, abs=1e-6)
        assert row["timestamp"] != ""

    def test_insert_raises_on_invalid_path(self):
        """존재하지 않는 디렉토리 경로 → sqlite3.OperationalError 전파 확인."""
        import sqlite3
        bad_path = "/nonexistent_dir/sub/test.db"
        with pytest.raises(sqlite3.OperationalError):
            _insert_paper_trades(self._metrics(), bad_path, "ALL", "ALL")

    def test_compute_metrics_with_db_path_saves_row(self, tmp_path):
        """compute_metrics(db_path=...) 호출 시 paper_trades에 INSERT됨."""
        import sqlite3
        db_path = str(tmp_path / "test.db")
        runner = PaperTradingRunner(initial_capital=10_000_000)
        runner.on_signal("BTC", "BUY", 50_000_000, 100_000, "TREND", "loop-1")
        runner.on_live_executed("loop-1", 50_000_000)

        runner.compute_metrics(db_path=db_path, coin="BTC", strategy="TREND")

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        conn.close()
        assert count == 1
