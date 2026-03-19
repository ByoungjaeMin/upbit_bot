"""
monitoring/dashboard.py — Streamlit 실시간 대시보드

실행: streamlit run monitoring/dashboard.py
     (upbit_bot/ 루트에서 실행 권장)

[화면 구성]
  상단  : 자본 / 오늘손익 / 승률 / 현재전략 메트릭 카드
  차트1 : 자본 곡선 (전체기간)
  차트2 : 오늘 거래 타임라인 (전략별 색상)
  차트3 : 전략별 기여도 + 동적 가중치
  차트4 : 앙상블 모델별 정확도
  차트5 : HMM 레짐 히스토리
  차트6 : 데이터 품질 점수 트렌드
  차트7 : 김치프리미엄 24h 추이
  차트8 : 전략별 롤링 샤프비율 (4주) + DORMANT 하이라이트
  차트9 : Monte Carlo 분포 (백테스트 결과 존재 시)
  차트10: 활성 페어리스트 거래량 순위
  테이블: 현재 포지션 목록
  테이블: 최근 20거래
  하단  : 디스크 사용량 게이지 + 다음 정리 예정시각

[설계]
  - 데이터 소스: SQLite (db_path는 환경변수 BOT_DB_PATH 또는 기본값)
  - 자동새로고침: st.rerun() + time.sleep(30)
  - localhost 전용 (Streamlit 기본)
  - pandas로 SQLite 쿼리 → plotly 차트
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Streamlit 임포트 (실행 시에만 활성)
# ─────────────────────────────────────────────────────────────────
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

REFRESH_SEC = 30
DB_PATH = Path(os.getenv("BOT_DB_PATH", "data/bot.db"))


# ─────────────────────────────────────────────────────────────────
# 데이터 로딩 헬퍼
# ─────────────────────────────────────────────────────────────────

def _connect(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def load_capital_curve(conn: sqlite3.Connection) -> pd.DataFrame:
    """전체기간 자본 곡선 — trades 테이블 누적."""
    df = _query(
        conn,
        """
        SELECT timestamp, pnl
        FROM trades
        WHERE is_dry_run = 0 AND side = 'SELL' AND pnl IS NOT NULL
        ORDER BY timestamp
        """,
    )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cumulative_pnl"] = df["pnl"].cumsum()
    return df


def load_today_trades(conn: sqlite3.Connection) -> pd.DataFrame:
    """오늘 거래 타임라인."""
    today = datetime.now(timezone.utc).date().isoformat()
    return _query(
        conn,
        """
        SELECT timestamp, coin, side, price, krw_amount, pnl, strategy_type
        FROM trades
        WHERE timestamp >= ? ORDER BY timestamp
        """,
        (today,),
    )


def load_strategy_contrib(conn: sqlite3.Connection) -> pd.DataFrame:
    """전략별 실현 손익 기여도."""
    return _query(
        conn,
        """
        SELECT strategy_type,
               SUM(pnl)            AS total_pnl,
               COUNT(*)            AS trade_count,
               AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) AS win_rate
        FROM trades
        WHERE side = 'SELL' AND pnl IS NOT NULL
        GROUP BY strategy_type
        """,
    )


def load_ensemble_accuracy(conn: sqlite3.Connection) -> pd.DataFrame:
    """앙상블 모델별 정확도 (최근 200개 예측 기준)."""
    df = _query(
        conn,
        """
        SELECT xgb_prob, lgb_prob, lstm_prob, gru_prob, signal_confirmed
        FROM ensemble_predictions
        ORDER BY timestamp DESC LIMIT 200
        """,
    )
    if df.empty:
        return df
    rows = []
    for col in ["xgb_prob", "lgb_prob", "lstm_prob", "gru_prob"]:
        if col in df.columns:
            acc = ((df[col] >= 0.5) == df["signal_confirmed"].astype(bool)).mean()
            rows.append({"model": col.replace("_prob", "").upper(), "accuracy": acc})
    return pd.DataFrame(rows)


def load_hmm_history(conn: sqlite3.Connection) -> pd.DataFrame:
    """HMM 레짐 히스토리."""
    return _query(
        conn,
        """
        SELECT timestamp, hmm_regime, weighted_avg
        FROM ensemble_predictions
        WHERE hmm_regime >= 0
        ORDER BY timestamp DESC LIMIT 500
        """,
    )


def load_kimchi(conn: sqlite3.Connection) -> pd.DataFrame:
    """김치프리미엄 최근 24h."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    return _query(
        conn,
        """
        SELECT timestamp, kimchi_premium_pct
        FROM kimchi_premium_log
        WHERE timestamp >= ?
        ORDER BY timestamp
        """,
        (cutoff,),
    )


def load_rolling_sharpe(conn: sqlite3.Connection) -> pd.DataFrame:
    """전략별 롤링 샤프비율 + DORMANT 상태."""
    return _query(
        conn,
        """
        SELECT week_start, strategy_type, rolling_sharpe, is_dormant
        FROM strategy_decay_log
        ORDER BY week_start DESC LIMIT 100
        """,
    )


def load_pairs(conn: sqlite3.Connection) -> pd.DataFrame:
    """최근 스캔 페어리스트."""
    latest_ts = _query(
        conn,
        "SELECT MAX(timestamp) AS ts FROM coin_scan_results",
    )
    if latest_ts.empty or latest_ts["ts"].iloc[0] is None:
        return pd.DataFrame()
    ts = latest_ts["ts"].iloc[0]
    return _query(
        conn,
        """
        SELECT coin, rank_by_volume, volume_24h_krw, included
        FROM coin_scan_results
        WHERE timestamp = ?
        ORDER BY rank_by_volume
        """,
        (ts,),
    )


def load_quality_trend(conn: sqlite3.Connection) -> pd.DataFrame:
    """데이터 품질 점수 트렌드 (layer1_log의 tradeable 비율로 대체)."""
    return _query(
        conn,
        """
        SELECT DATE(timestamp) AS date,
               AVG(CAST(tradeable AS FLOAT)) AS quality_score
        FROM layer1_log
        GROUP BY DATE(timestamp)
        ORDER BY date DESC LIMIT 30
        """,
    )


def load_open_positions(conn: sqlite3.Connection) -> pd.DataFrame:
    """현재 보유 포지션 (BUY 후 SELL 미체결)."""
    return _query(
        conn,
        """
        SELECT coin,
               timestamp AS entry_time,
               price     AS entry_price,
               krw_amount,
               strategy_type
        FROM trades
        WHERE side = 'BUY' AND is_dry_run = 0
          AND coin NOT IN (
            SELECT coin FROM trades WHERE side = 'SELL' AND is_dry_run = 0
          )
        ORDER BY timestamp DESC
        """,
    )


def load_recent_trades(conn: sqlite3.Connection, n: int = 20) -> pd.DataFrame:
    """최근 N건 거래."""
    return _query(
        conn,
        f"""
        SELECT timestamp, coin, side, price, krw_amount, pnl, pnl_pct, strategy_type
        FROM trades
        ORDER BY timestamp DESC LIMIT {n}
        """,
    )


def load_disk_stats(db_path: Path) -> dict[str, float]:
    import shutil
    disk = shutil.disk_usage(str(db_path.parent))
    return {
        "free_gb": disk.free / (1024 ** 3),
        "total_gb": disk.total / (1024 ** 3),
        "db_mb": db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0.0,
    }


# ─────────────────────────────────────────────────────────────────
# 요약 메트릭 계산
# ─────────────────────────────────────────────────────────────────

def compute_summary(today_trades: pd.DataFrame, all_trades: pd.DataFrame) -> dict:
    today_pnl = today_trades["pnl"].sum() if "pnl" in today_trades.columns else 0.0
    sells = all_trades[all_trades["side"] == "SELL"] if not all_trades.empty and "side" in all_trades.columns else pd.DataFrame()
    win_rate = (sells["pnl"] > 0).mean() if not sells.empty and "pnl" in sells.columns else 0.0
    capital = all_trades["krw_amount"].sum() if not all_trades.empty and "krw_amount" in all_trades.columns else 0.0
    return {
        "today_pnl": float(today_pnl or 0),
        "win_rate": float(win_rate or 0),
        "capital": float(capital or 0),
    }


# ─────────────────────────────────────────────────────────────────
# Streamlit 대시보드 메인
# ─────────────────────────────────────────────────────────────────

def run_dashboard(db_path: Path = DB_PATH) -> None:
    """Streamlit 대시보드 진입점."""
    st.set_page_config(
        page_title="업비트 퀀트봇 대시보드",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("📈 업비트 퀀트봇 대시보드")
    st.caption(f"마지막 새로고침: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    conn = _connect(db_path)
    if conn is None:
        st.error(f"DB 연결 실패: {db_path} — 봇을 먼저 실행하세요.")
        time.sleep(REFRESH_SEC)
        st.rerun()
        return

    try:
        _render(conn, db_path)
    finally:
        conn.close()

    # 자동 새로고침
    time.sleep(REFRESH_SEC)
    st.rerun()


def _render(conn: sqlite3.Connection, db_path: Path) -> None:
    """실제 렌더링 로직 (테스트 분리용)."""

    # ── 데이터 로딩 ──────────────────────────────────────────────
    today_trades   = load_today_trades(conn)
    all_trades_df  = _query(conn, "SELECT * FROM trades ORDER BY timestamp")
    capital_curve  = load_capital_curve(conn)
    strategy_contrib = load_strategy_contrib(conn)
    ensemble_acc   = load_ensemble_accuracy(conn)
    hmm_history    = load_hmm_history(conn)
    quality_trend  = load_quality_trend(conn)
    kimchi_df      = load_kimchi(conn)
    rolling_sharpe = load_rolling_sharpe(conn)
    pairs_df       = load_pairs(conn)
    open_positions = load_open_positions(conn)
    recent_trades  = load_recent_trades(conn)
    disk_stats     = load_disk_stats(db_path)
    summary        = compute_summary(today_trades, all_trades_df)

    # ── 상단 메트릭 카드 ─────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 총 투입자본", f"{summary['capital']:,.0f}원")
    pnl_sign = "+" if summary["today_pnl"] >= 0 else ""
    col2.metric("📊 오늘 실현손익", f"{pnl_sign}{summary['today_pnl']:,.0f}원")
    col3.metric("🎯 전체 승률", f"{summary['win_rate']:.1%}")
    col4.metric("💾 DB 크기", f"{disk_stats['db_mb']:.1f} MB")

    st.divider()

    # ── 차트1: 자본 곡선 ─────────────────────────────────────────
    st.subheader("차트1 · 자본 곡선 (전체기간)")
    if not capital_curve.empty:
        fig = px.line(
            capital_curve, x="timestamp", y="cumulative_pnl",
            labels={"cumulative_pnl": "누적 손익 (원)", "timestamp": "시각"},
            color_discrete_sequence=["#00c8ff"],
        )
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("거래 데이터 없음")

    # ── 차트2: 오늘 거래 타임라인 ────────────────────────────────
    st.subheader("차트2 · 오늘 거래 타임라인")
    if not today_trades.empty and "timestamp" in today_trades.columns:
        today_trades["timestamp"] = pd.to_datetime(today_trades["timestamp"])
        fig2 = px.scatter(
            today_trades, x="timestamp", y="coin",
            color="strategy_type", size_max=12,
            symbol="side",
            hover_data=["price", "pnl"],
            labels={"timestamp": "시각", "coin": "코인"},
        )
        fig2.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("오늘 거래 없음")

    # ── 차트3: 전략별 기여도 ─────────────────────────────────────
    st.subheader("차트3 · 전략별 기여도 + 동적 가중치")
    if not strategy_contrib.empty:
        fig3 = px.bar(
            strategy_contrib, x="strategy_type", y="total_pnl",
            color="strategy_type",
            labels={"total_pnl": "실현손익 (원)", "strategy_type": "전략"},
            text_auto=True,
        )
        fig3.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("전략 데이터 없음")

    # ── 차트4: 앙상블 모델별 정확도 ──────────────────────────────
    st.subheader("차트4 · 앙상블 모델별 정확도")
    if not ensemble_acc.empty:
        fig4 = px.bar(
            ensemble_acc, x="model", y="accuracy",
            color="model", text_auto=".1%",
            range_y=[0, 1],
            labels={"accuracy": "정확도", "model": "모델"},
        )
        fig4.update_layout(height=280, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("앙상블 예측 데이터 없음")

    # ── 차트5: HMM 레짐 히스토리 ─────────────────────────────────
    st.subheader("차트5 · HMM 레짐 히스토리")
    if not hmm_history.empty:
        hmm_history["timestamp"] = pd.to_datetime(hmm_history["timestamp"])
        regime_labels = {0: "강한상승", 1: "약상승/횡보", 2: "약하락/횡보", 3: "강한하락"}
        hmm_history["레짐"] = hmm_history["hmm_regime"].map(regime_labels)
        fig5 = px.scatter(
            hmm_history, x="timestamp", y="레짐",
            color="레짐",
            color_discrete_map={
                "강한상승": "green", "약상승/횡보": "lightgreen",
                "약하락/횡보": "orange", "강한하락": "red",
            },
        )
        fig5.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("HMM 데이터 없음 (Phase C 이후 활성화)")

    # ── 차트6: 데이터 품질 점수 트렌드 ───────────────────────────
    st.subheader("차트6 · 데이터 품질 점수 트렌드")
    if not quality_trend.empty:
        fig6 = px.line(
            quality_trend.sort_values("date"), x="date", y="quality_score",
            labels={"quality_score": "품질 점수 (tradeable 비율)", "date": "날짜"},
            color_discrete_sequence=["#ffa500"],
        )
        fig6.update_yaxes(range=[0, 1])
        fig6.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("품질 데이터 없음")

    # ── 차트7: 김치프리미엄 24h 추이 ─────────────────────────────
    st.subheader("차트7 · 김치프리미엄 24h 추이")
    if not kimchi_df.empty:
        kimchi_df["timestamp"] = pd.to_datetime(kimchi_df["timestamp"])
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(
            x=kimchi_df["timestamp"], y=kimchi_df["kimchi_premium_pct"],
            mode="lines", name="김치프리미엄",
            line=dict(color="#ff6b6b"),
        ))
        # 과열(+5%) / 역프리미엄(-1%) 경계선
        fig7.add_hline(y=5.0,  line_dash="dash", line_color="red",
                       annotation_text="과열 +5%")
        fig7.add_hline(y=-1.0, line_dash="dash", line_color="blue",
                       annotation_text="역프리미엄 -1%")
        fig7.add_hline(y=0,    line_color="gray", line_width=0.5)
        fig7.update_layout(height=280, margin=dict(t=10, b=10),
                           yaxis_title="김치프리미엄 (%)")
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("김치프리미엄 데이터 없음")

    # ── 차트8: 전략별 롤링 샤프비율 ──────────────────────────────
    st.subheader("차트8 · 전략별 롤링 샤프비율 (4주) + DORMANT 하이라이트")
    if not rolling_sharpe.empty:
        rolling_sharpe["week_start"] = pd.to_datetime(rolling_sharpe["week_start"])
        fig8 = px.line(
            rolling_sharpe, x="week_start", y="rolling_sharpe",
            color="strategy_type",
            labels={"rolling_sharpe": "롤링 샤프비율", "week_start": "주"},
        )
        # DORMANT 전략 강조
        dormant = rolling_sharpe[rolling_sharpe["is_dormant"] == 1]
        if not dormant.empty:
            fig8.add_trace(go.Scatter(
                x=dormant["week_start"], y=dormant["rolling_sharpe"],
                mode="markers", marker=dict(color="red", size=10, symbol="x"),
                name="DORMANT",
            ))
        fig8.add_hline(y=0, line_dash="dash", line_color="gray")
        fig8.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("전략 decay 데이터 없음")

    # ── 차트9: Monte Carlo (백테스트 결과 있을 때) ───────────────
    st.subheader("차트9 · Monte Carlo 분포")
    mc_results_dir = Path("backtest/results")
    mc_files = sorted(mc_results_dir.glob("montecarlo_*.json"), reverse=True) \
        if mc_results_dir.exists() else []
    if mc_files:
        import json
        try:
            with open(mc_files[0]) as f:
                mc_data = json.load(f)
            sharpe_dist = mc_data.get("sharpe_distribution", [])
            actual_sharpe = mc_data.get("actual_sharpe", 0)
            p_value = mc_data.get("p_value", 1.0)
            if sharpe_dist:
                fig9 = go.Figure()
                fig9.add_trace(go.Histogram(
                    x=sharpe_dist, nbinsx=50,
                    name="MC 샤프 분포",
                    marker_color="steelblue",
                ))
                fig9.add_vline(
                    x=actual_sharpe, line_color="red", line_dash="dash",
                    annotation_text=f"실제 샤프={actual_sharpe:.3f} (p={p_value:.3f})",
                )
                fig9.update_layout(height=280, margin=dict(t=10, b=10),
                                   xaxis_title="샤프비율", yaxis_title="빈도")
                st.plotly_chart(fig9, use_container_width=True)
                if p_value < 0.05:
                    st.success(f"✅ 통계적 유의성 확인 (p={p_value:.4f} < 0.05)")
                else:
                    st.warning(f"⚠️ 통계적 유의성 미확보 (p={p_value:.4f} ≥ 0.05)")
        except Exception:
            st.info("Monte Carlo 결과 파일 파싱 오류")
    else:
        st.info("Monte Carlo 결과 없음 (Phase B 백테스트 후 활성화)")

    # ── 차트10: 활성 페어리스트 ───────────────────────────────────
    st.subheader("차트10 · 활성 페어리스트 (거래량 순위)")
    if not pairs_df.empty:
        active = pairs_df[pairs_df["included"] == 1].head(30)
        if not active.empty:
            fig10 = px.bar(
                active, x="coin", y="volume_24h_krw",
                color="rank_by_volume",
                color_continuous_scale="Blues_r",
                labels={"volume_24h_krw": "24h 거래량 (KRW)", "coin": "코인"},
            )
            fig10.update_layout(height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig10, use_container_width=True)
        else:
            st.info("활성 페어 없음")
    else:
        st.info("페어리스트 스캔 데이터 없음")

    st.divider()

    # ── 테이블: 현재 포지션 ───────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("현재 포지션")
        if not open_positions.empty:
            st.dataframe(open_positions, use_container_width=True, hide_index=True)
        else:
            st.info("보유 포지션 없음")

    with col_r:
        st.subheader("최근 20거래")
        if not recent_trades.empty:
            def _color_pnl(val: float) -> str:
                if not isinstance(val, (int, float)):
                    return ""
                return "color: green" if val >= 0 else "color: red"
            styled = recent_trades.style.applymap(_color_pnl, subset=["pnl"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("거래 내역 없음")

    st.divider()

    # ── 하단: 디스크 사용량 게이지 ───────────────────────────────
    st.subheader("디스크 사용량")
    col_d1, col_d2, col_d3 = st.columns(3)
    used_pct = 1 - disk_stats["free_gb"] / max(disk_stats["total_gb"], 1)
    col_d1.metric("여유 공간", f"{disk_stats['free_gb']:.1f} GB")
    col_d2.metric("총 용량", f"{disk_stats['total_gb']:.1f} GB")
    col_d3.metric("DB 크기", f"{disk_stats['db_mb']:.1f} MB")

    fig_disk = go.Figure(go.Indicator(
        mode="gauge+number",
        value=used_pct * 100,
        number={"suffix": "%"},
        title={"text": "디스크 사용률"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "steelblue"},
            "steps": [
                {"range": [0, 60],  "color": "lightgreen"},
                {"range": [60, 80], "color": "yellow"},
                {"range": [80, 100],"color": "red"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90,
            },
        },
    ))
    fig_disk.update_layout(height=220, margin=dict(t=20, b=0))
    st.plotly_chart(fig_disk, use_container_width=True)

    # 다음 정리 예정시각
    now = datetime.now(timezone.utc)
    next_daily = now.replace(hour=3, minute=0, second=0, microsecond=0)
    if now >= next_daily:
        next_daily += timedelta(days=1)
    st.caption(f"다음 정기 정리 예정: {next_daily.strftime('%Y-%m-%d 03:00 UTC')}")


# ─────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__" or (
    _STREAMLIT_AVAILABLE and hasattr(__builtins__, "__STREAMLIT_SERVER__")
):
    if _STREAMLIT_AVAILABLE:
        run_dashboard()
