"""test_dashboard.py — dashboard.py 데이터 로딩 함수 단위 테스트.

Streamlit UI 렌더링은 테스트하지 않음.
데이터 쿼리 함수(load_*)와 compute_summary만 검증.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

# Streamlit + plotly 미설치 환경 대비 mock
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("streamlit", MagicMock())
sys.modules.setdefault("plotly", MagicMock())
sys.modules.setdefault("plotly.express", MagicMock())
sys.modules.setdefault("plotly.graph_objects", MagicMock())

from upbit_bot.monitoring.dashboard import (  # noqa: E402
    compute_summary,
    load_capital_curve,
    load_disk_stats,
    load_ensemble_accuracy,
    load_hmm_history,
    load_kimchi,
    load_open_positions,
    load_pairs,
    load_quality_trend,
    load_recent_trades,
    load_rolling_sharpe,
    load_strategy_contrib,
    load_today_trades,
)


# ─────────────────────────────────────────────────────────────────
# 픽스처
# ─────────────────────────────────────────────────────────────────

def _make_db(tmp_path: Path) -> tuple[Path, sqlite3.Connection]:
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT, timestamp TEXT, side TEXT,
            price REAL, krw_amount REAL, pnl REAL, pnl_pct REAL,
            strategy_type TEXT, is_dry_run INTEGER DEFAULT 0,
            order_id TEXT, paper_trade INTEGER DEFAULT 0,
            action INTEGER DEFAULT 0, volume REAL DEFAULT 0,
            fee REAL DEFAULT 0, slippage REAL DEFAULT 0,
            kelly_f REAL, position_size REAL
        );
        CREATE TABLE ensemble_predictions (
            id INTEGER PRIMARY KEY,
            coin TEXT, timestamp TEXT,
            xgb_prob REAL, lgb_prob REAL, lstm_prob REAL, gru_prob REAL,
            weighted_avg REAL DEFAULT 0,
            consensus_count INTEGER DEFAULT 0,
            signal_confirmed INTEGER DEFAULT 0,
            hmm_regime INTEGER DEFAULT -1,
            hmm_confidence REAL DEFAULT 0
        );
        CREATE TABLE layer1_log (
            id INTEGER PRIMARY KEY,
            coin TEXT, timestamp TEXT, tradeable INTEGER DEFAULT 1,
            regime_strategy TEXT
        );
        CREATE TABLE kimchi_premium_log (
            id INTEGER PRIMARY KEY,
            timestamp TEXT UNIQUE,
            upbit_btc_krw REAL, binance_btc_usd REAL,
            usd_krw_rate REAL, kimchi_premium_pct REAL
        );
        CREATE TABLE strategy_decay_log (
            id INTEGER PRIMARY KEY,
            week_start TEXT, strategy_type TEXT,
            rolling_sharpe REAL, win_rate REAL,
            profit_loss_ratio REAL, trade_count INTEGER DEFAULT 0,
            is_dormant INTEGER DEFAULT 0,
            dormant_since TEXT, revival_date TEXT
        );
        CREATE TABLE coin_scan_results (
            id INTEGER PRIMARY KEY,
            timestamp TEXT, coin TEXT,
            rank_by_volume INTEGER, volume_24h_krw REAL,
            is_leverage_token INTEGER DEFAULT 0,
            is_blacklisted INTEGER DEFAULT 0,
            included INTEGER NOT NULL, reason_excluded TEXT
        );
    """)
    conn.commit()
    return db, conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


def _insert_trade(conn: sqlite3.Connection, side: str = "SELL",
                  pnl: float = 1000.0, strategy: str = "TREND_STRONG",
                  ts: str | None = None) -> None:
    conn.execute(
        """INSERT INTO trades (coin, timestamp, side, price, krw_amount, pnl,
               strategy_type, is_dry_run)
           VALUES (?, ?, ?, 1000000, 50000, ?, ?, 0)""",
        ("KRW-BTC", ts or _now(), side, pnl, strategy),
    )
    conn.commit()


@pytest.fixture
def db_conn(tmp_path: Path):
    db, conn = _make_db(tmp_path)
    yield db, conn
    conn.close()


# ─────────────────────────────────────────────────────────────────
# load_capital_curve
# ─────────────────────────────────────────────────────────────────

class TestLoadCapitalCurve:
    def test_empty_returns_empty(self, db_conn) -> None:
        _, conn = db_conn
        df = load_capital_curve(conn)
        assert df.empty

    def test_cumulative_pnl(self, db_conn) -> None:
        _, conn = db_conn
        _insert_trade(conn, side="SELL", pnl=1000.0)
        _insert_trade(conn, side="SELL", pnl=2000.0)
        df = load_capital_curve(conn)
        assert not df.empty
        assert "cumulative_pnl" in df.columns
        assert df["cumulative_pnl"].iloc[-1] == pytest.approx(3000.0)

    def test_excludes_buy_side(self, db_conn) -> None:
        _, conn = db_conn
        _insert_trade(conn, side="BUY", pnl=None)
        df = load_capital_curve(conn)
        assert df.empty

    def test_excludes_dry_run(self, db_conn) -> None:
        _, conn = db_conn
        conn.execute(
            "INSERT INTO trades (coin, timestamp, side, price, krw_amount, pnl, is_dry_run)"
            " VALUES ('KRW-ETH', ?, 'SELL', 1, 50000, 500, 1)",
            (_now(),),
        )
        conn.commit()
        df = load_capital_curve(conn)
        assert df.empty


# ─────────────────────────────────────────────────────────────────
# load_today_trades
# ─────────────────────────────────────────────────────────────────

class TestLoadTodayTrades:
    def test_returns_today_only(self, db_conn) -> None:
        _, conn = db_conn
        _insert_trade(conn, ts=_now())
        _insert_trade(conn, ts=_days_ago(2))

        df = load_today_trades(conn)
        assert len(df) == 1

    def test_empty_when_no_trades_today(self, db_conn) -> None:
        _, conn = db_conn
        _insert_trade(conn, ts=_days_ago(2))
        df = load_today_trades(conn)
        assert df.empty


# ─────────────────────────────────────────────────────────────────
# load_strategy_contrib
# ─────────────────────────────────────────────────────────────────

class TestLoadStrategyContrib:
    def test_groups_by_strategy(self, db_conn) -> None:
        _, conn = db_conn
        _insert_trade(conn, strategy="TREND_STRONG", pnl=1000)
        _insert_trade(conn, strategy="TREND_STRONG", pnl=2000)
        _insert_trade(conn, strategy="GRID", pnl=-500)

        df = load_strategy_contrib(conn)
        assert not df.empty
        trend = df[df["strategy_type"] == "TREND_STRONG"]
        assert len(trend) == 1
        assert trend["total_pnl"].iloc[0] == pytest.approx(3000.0)

    def test_empty_returns_empty(self, db_conn) -> None:
        _, conn = db_conn
        df = load_strategy_contrib(conn)
        assert df.empty


# ─────────────────────────────────────────────────────────────────
# load_ensemble_accuracy
# ─────────────────────────────────────────────────────────────────

class TestLoadEnsembleAccuracy:
    def test_empty_returns_empty(self, db_conn) -> None:
        _, conn = db_conn
        df = load_ensemble_accuracy(conn)
        assert df.empty

    def test_accuracy_range(self, db_conn) -> None:
        _, conn = db_conn
        conn.execute(
            """INSERT INTO ensemble_predictions
               (coin, timestamp, xgb_prob, lgb_prob, weighted_avg,
                consensus_count, signal_confirmed)
               VALUES (?, ?, 0.7, 0.65, 0.68, 2, 1)""",
            ("KRW-BTC", _now()),
        )
        conn.commit()
        df = load_ensemble_accuracy(conn)
        assert not df.empty
        assert all(0 <= acc <= 1 for acc in df["accuracy"])


# ─────────────────────────────────────────────────────────────────
# load_hmm_history
# ─────────────────────────────────────────────────────────────────

class TestLoadHmmHistory:
    def test_empty_when_no_hmm(self, db_conn) -> None:
        _, conn = db_conn
        conn.execute(
            "INSERT INTO ensemble_predictions (coin, timestamp, weighted_avg,"
            " consensus_count, signal_confirmed, hmm_regime)"
            " VALUES ('KRW-BTC', ?, 0.6, 2, 1, -1)",
            (_now(),),
        )
        conn.commit()
        df = load_hmm_history(conn)
        assert df.empty

    def test_returns_regime_data(self, db_conn) -> None:
        _, conn = db_conn
        conn.execute(
            "INSERT INTO ensemble_predictions (coin, timestamp, weighted_avg,"
            " consensus_count, signal_confirmed, hmm_regime)"
            " VALUES ('KRW-BTC', ?, 0.6, 2, 1, 1)",
            (_now(),),
        )
        conn.commit()
        df = load_hmm_history(conn)
        assert not df.empty
        assert "hmm_regime" in df.columns


# ─────────────────────────────────────────────────────────────────
# load_kimchi
# ─────────────────────────────────────────────────────────────────

class TestLoadKimchi:
    def test_empty_returns_empty(self, db_conn) -> None:
        _, conn = db_conn
        df = load_kimchi(conn)
        assert df.empty

    def test_returns_24h_data(self, db_conn) -> None:
        _, conn = db_conn
        conn.execute(
            "INSERT INTO kimchi_premium_log"
            " (timestamp, upbit_btc_krw, binance_btc_usd, usd_krw_rate, kimchi_premium_pct)"
            " VALUES (?, 100000000, 75000, 1330, 2.5)",
            (_now(),),
        )
        conn.commit()
        df = load_kimchi(conn)
        assert not df.empty
        assert "kimchi_premium_pct" in df.columns


# ─────────────────────────────────────────────────────────────────
# load_quality_trend
# ─────────────────────────────────────────────────────────────────

class TestLoadQualityTrend:
    def test_empty_returns_empty(self, db_conn) -> None:
        _, conn = db_conn
        df = load_quality_trend(conn)
        assert df.empty

    def test_quality_score_range(self, db_conn) -> None:
        _, conn = db_conn
        for tradeable in [1, 1, 0]:
            conn.execute(
                "INSERT INTO layer1_log (coin, timestamp, tradeable)"
                " VALUES ('KRW-BTC', ?, ?)",
                (_now(), tradeable),
            )
        conn.commit()
        df = load_quality_trend(conn)
        assert not df.empty
        assert all(0 <= v <= 1 for v in df["quality_score"])


# ─────────────────────────────────────────────────────────────────
# load_rolling_sharpe
# ─────────────────────────────────────────────────────────────────

class TestLoadRollingSharpe:
    def test_empty_returns_empty(self, db_conn) -> None:
        _, conn = db_conn
        df = load_rolling_sharpe(conn)
        assert df.empty

    def test_returns_decay_data(self, db_conn) -> None:
        _, conn = db_conn
        conn.execute(
            "INSERT INTO strategy_decay_log"
            " (week_start, strategy_type, rolling_sharpe, is_dormant)"
            " VALUES ('2026-03-11', 'TREND_STRONG', 1.5, 0)",
        )
        conn.commit()
        df = load_rolling_sharpe(conn)
        assert not df.empty
        assert "rolling_sharpe" in df.columns


# ─────────────────────────────────────────────────────────────────
# load_pairs
# ─────────────────────────────────────────────────────────────────

class TestLoadPairs:
    def test_empty_returns_empty(self, db_conn) -> None:
        _, conn = db_conn
        df = load_pairs(conn)
        assert df.empty

    def test_returns_latest_scan(self, db_conn) -> None:
        _, conn = db_conn
        ts = _now()
        conn.execute(
            "INSERT INTO coin_scan_results"
            " (timestamp, coin, rank_by_volume, volume_24h_krw, included)"
            " VALUES (?, 'KRW-BTC', 1, 1e10, 1)",
            (ts,),
        )
        conn.commit()
        df = load_pairs(conn)
        assert not df.empty
        assert df["coin"].iloc[0] == "KRW-BTC"


# ─────────────────────────────────────────────────────────────────
# load_open_positions & load_recent_trades
# ─────────────────────────────────────────────────────────────────

class TestLoadPositionsAndTrades:
    def test_open_positions_empty(self, db_conn) -> None:
        _, conn = db_conn
        df = load_open_positions(conn)
        assert isinstance(df, pd.DataFrame)

    def test_recent_trades_empty(self, db_conn) -> None:
        _, conn = db_conn
        df = load_recent_trades(conn)
        assert isinstance(df, pd.DataFrame)

    def test_recent_trades_limit(self, db_conn) -> None:
        _, conn = db_conn
        for _ in range(25):
            _insert_trade(conn)
        df = load_recent_trades(conn, n=20)
        assert len(df) == 20


# ─────────────────────────────────────────────────────────────────
# load_disk_stats
# ─────────────────────────────────────────────────────────────────

class TestLoadDiskStats:
    def test_returns_dict(self, tmp_path: Path) -> None:
        db = tmp_path / "bot.db"
        db.write_bytes(b"x" * 1024)
        stats = load_disk_stats(db)
        assert "free_gb" in stats
        assert "db_mb" in stats
        assert stats["db_mb"] > 0

    def test_nonexistent_db_zero_size(self, tmp_path: Path) -> None:
        stats = load_disk_stats(tmp_path / "nonexistent.db")
        assert stats["db_mb"] == 0.0


# ─────────────────────────────────────────────────────────────────
# compute_summary
# ─────────────────────────────────────────────────────────────────

class TestComputeSummary:
    def test_empty_dfs(self) -> None:
        summary = compute_summary(pd.DataFrame(), pd.DataFrame())
        assert summary["today_pnl"] == pytest.approx(0.0)
        assert summary["win_rate"] == pytest.approx(0.0)
        assert summary["capital"] == pytest.approx(0.0)

    def test_today_pnl(self) -> None:
        today = pd.DataFrame({"pnl": [1000.0, 2000.0]})
        summary = compute_summary(today, pd.DataFrame())
        assert summary["today_pnl"] == pytest.approx(3000.0)

    def test_win_rate(self) -> None:
        all_t = pd.DataFrame({
            "side": ["SELL", "SELL", "SELL", "SELL"],
            "pnl": [100, 200, -50, -100],
            "krw_amount": [50000, 50000, 50000, 50000],
        })
        summary = compute_summary(pd.DataFrame(), all_t)
        assert summary["win_rate"] == pytest.approx(0.5)
