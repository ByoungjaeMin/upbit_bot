"""
data/candle_builder.py — WebSocket 체결 이벤트 → OHLCV 캔들 합성

Phase A 핵심: REST 90회/루프 → 0회 대체.

책임:
  - trade 이벤트 수신 → 5분 버킷 누적
  - 5분 경계 도달 → 완성 캔들 확정 + pandas-ta 지표 계산
  - 12개 5분봉 → 1시간봉 자동 합성
  - Silent-drop 감지: 체결 < 10건 → REST 폴백
  - 마이크로스트럭처 피처: tick_imbalance, trade_velocity
  - 일봉 지표는 daily_update() 에서만 처리 (shift(1) 필수)
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable

import numpy as np
import pandas as pd
import pandas_ta as ta

from schema import MarketState, RawMarketData

logger = logging.getLogger(__name__)

CANDLE_5M_SECONDS = 300          # 5분 = 300초
MIN_TRADES_PER_CANDLE = 10       # 미만이면 silent-drop 의심
MAX_CANDLES_IN_MEMORY = 500      # 코인당 인메모리 최대 캔들 수


# ---------------------------------------------------------------------------
# 5분 버킷 (체결 누적)
# ---------------------------------------------------------------------------

@dataclass
class _TradeBucket:
    """5분 단위 체결 데이터 누적."""
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = float("inf")
    close_price: float = 0.0
    volume: float = 0.0
    buy_volume: float = 0.0        # ask_bid == 'ASK'
    sell_volume: float = 0.0       # ask_bid == 'BID'
    trade_count: int = 0
    bucket_start: datetime | None = None
    # 체결 속도 계산용 (30초 슬라이딩 윈도우)
    _recent_ts: list[datetime] = field(default_factory=list)

    def is_empty(self) -> bool:
        return self.trade_count == 0

    def update(self, trade: RawMarketData) -> None:
        price = trade.trade_price
        vol = trade.trade_volume
        ts = trade.timestamp

        if self.trade_count == 0:
            self.open_price = price
            self.high_price = price
            self.low_price = price
            self.bucket_start = ts

        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.close_price = price
        self.volume += vol
        self.trade_count += 1
        self._recent_ts.append(ts)

        if trade.ask_bid == "ASK":
            self.buy_volume += vol
        else:
            self.sell_volume += vol

    def tick_imbalance(self) -> float:
        """(매수체결량 - 매도체결량) / 전체체결량 → -1~+1."""
        total = self.buy_volume + self.sell_volume
        if total == 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / total

    def trade_velocity(self, bucket_close: datetime) -> float:
        """최근 30초 체결 건수 / 직전 30초 체결 건수 (가속도).

        bucket_close: 이 버킷이 닫히는 시각 (= 다음 5분 경계)
        """
        if not self._recent_ts:
            return 1.0
        recent_cutoff = bucket_close.timestamp() - 30
        prev_cutoff = recent_cutoff - 30
        recent = sum(1 for t in self._recent_ts if t.timestamp() >= recent_cutoff)
        prev = sum(1 for t in self._recent_ts if prev_cutoff <= t.timestamp() < recent_cutoff)
        if prev == 0:
            return float(recent) if recent > 0 else 1.0
        return recent / prev

    def to_ohlcv(self) -> dict:
        low = self.low_price if self.low_price != float("inf") else self.open_price
        return {
            "open": self.open_price,
            "high": self.high_price,
            "low": low,
            "close": self.close_price,
            "volume": self.volume,
        }


# ---------------------------------------------------------------------------
# CandleBuilder
# ---------------------------------------------------------------------------

class CandleBuilder:
    """WebSocket trade 이벤트 → OHLCV 캔들 합성기.

    콜백 (모두 Optional):
        on_candle_5m(coin, df_5m, micro_feats)  — 5분봉 완성 시
        on_candle_1h(coin, df_1h)               — 1시간봉 완성 시
        rest_fallback(coin, interval, gap_ts)   — 갭 발생 시 REST 보완
    """

    def __init__(
        self,
        cache=None,
        on_candle_5m: Callable[[str, pd.DataFrame, dict], Awaitable[None]] | None = None,
        on_candle_1h: Callable[[str, pd.DataFrame], Awaitable[None]] | None = None,
        rest_fallback: Callable[
            [str, str, datetime], Awaitable[pd.DataFrame | None]
        ] | None = None,
    ) -> None:
        self._cache = cache
        self._on_candle_5m = on_candle_5m
        self._on_candle_1h = on_candle_1h
        self._rest_fallback = rest_fallback

        # 코인별 현재 5분 버킷
        self._buckets: dict[str, _TradeBucket] = defaultdict(_TradeBucket)
        # 코인별 현재 버킷의 5분 경계 시각
        self._bucket_ts: dict[str, datetime] = {}
        # 코인별 완성 5분봉 (인메모리 최대 500개)
        self._candles_5m: dict[str, deque[dict]] = defaultdict(
            lambda: deque(maxlen=MAX_CANDLES_IN_MEMORY)
        )
        # 코인별 완성 1시간봉
        self._candles_1h: dict[str, deque[dict]] = defaultdict(
            lambda: deque(maxlen=MAX_CANDLES_IN_MEMORY)
        )
        # 1시간봉 합산용 임시 5분봉 누적 — 12개(=1시간) 초과분 자동 폐기
        self._pending_5m_for_1h: dict[str, deque[dict]] = defaultdict(
            lambda: deque(maxlen=12)
        )
        # silent-drop 감지용 마지막 trade 시각
        self._last_trade_ts: dict[str, datetime] = {}
        # 코인별 최신 일봉 피처 (shift(1) 적용) — daily_update() 호출 시 갱신
        # get_market_state_snapshot()에서 읽어 MarketState에 포함
        self._daily_features: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # 공개 헬퍼 — 초기 히스토리 로딩
    # ------------------------------------------------------------------

    def add_historical_candle(self, coin: str, row: dict) -> None:
        """REST 초기 히스토리 로딩 시 인메모리 캔들 deque에 추가.

        collector._initial_history_load()가 private attribute를 직접 조작하지 않도록
        이 메서드를 경유한다.
        """
        self._candles_5m[coin].append(row)

    # ------------------------------------------------------------------
    # 진입점: WebSocket trade 이벤트
    # ------------------------------------------------------------------

    async def on_trade(self, trade: RawMarketData) -> None:
        """WebSocket trade 이벤트 처리."""
        coin = trade.coin
        ts = trade.timestamp
        self._last_trade_ts[coin] = ts

        current_bucket_ts = self._floor_5m(ts)

        # 5분 경계 넘으면 이전 버킷 완성
        prev = self._bucket_ts.get(coin)
        if prev is not None and prev != current_bucket_ts:
            await self._finalize_candle(coin, prev)

        self._bucket_ts[coin] = current_bucket_ts
        self._buckets[coin].update(trade)

    # ------------------------------------------------------------------
    # 5분봉 완성
    # ------------------------------------------------------------------

    async def _finalize_candle(self, coin: str, bucket_ts: datetime) -> None:
        """버킷 → 완성 5분봉."""
        bucket = self._buckets[coin]
        next_boundary = datetime.fromtimestamp(
            bucket_ts.timestamp() + CANDLE_5M_SECONDS, tz=timezone.utc
        )

        if bucket.is_empty():
            logger.warning("[%s] 빈 버킷 %s — REST 폴백", coin, bucket_ts)
            await self._rest_gap_fill(coin, bucket_ts)
            self._buckets[coin] = _TradeBucket()
            return

        # silent-drop 감지
        if bucket.trade_count < MIN_TRADES_PER_CANDLE:
            logger.warning(
                "[%s] Silent-drop 의심 (%d건) — REST 폴백", coin, bucket.trade_count
            )
            await self._rest_gap_fill(coin, bucket_ts)

        micro = {
            "tick_imbalance": bucket.tick_imbalance(),
            "trade_velocity": bucket.trade_velocity(next_boundary),
        }

        candle: dict = {
            "coin": coin,
            "timestamp": bucket_ts.isoformat(),
            **bucket.to_ohlcv(),
            "tick_imbalance": micro["tick_imbalance"],
            "trade_velocity": micro["trade_velocity"],
        }

        self._candles_5m[coin].append(candle)
        self._pending_5m_for_1h[coin].append(candle)

        # SQLite 저장 (완성된 캔들만)
        if self._cache:
            self._cache.upsert_candle("candles_5m", coin, candle)

        # 지표 계산 후 콜백 — pandas-ta CPU 작업을 스레드 풀로 격리 (이벤트 루프 블로킹 방지)
        loop = asyncio.get_event_loop()
        df_5m = await loop.run_in_executor(None, self._build_df_with_indicators, coin, "5m")
        if self._on_candle_5m and df_5m is not None:
            await self._on_candle_5m(coin, df_5m, micro)

        # 1시간봉: 5분봉 12개 누적 시 합성
        if len(self._pending_5m_for_1h[coin]) >= 12:
            await self._finalize_1h_candle(coin)

        self._buckets[coin] = _TradeBucket()

    # ------------------------------------------------------------------
    # 1시간봉 합성
    # ------------------------------------------------------------------

    async def _finalize_1h_candle(self, coin: str) -> None:
        """12개 5분봉 → 1시간봉."""
        # deque(maxlen=12) 이므로 최대 12개 — 전체 스냅샷 후 초기화
        five_m = list(self._pending_5m_for_1h[coin])
        self._pending_5m_for_1h[coin].clear()

        h1: dict = {
            "coin": coin,
            "timestamp": five_m[0]["timestamp"],
            "open": five_m[0]["open"],
            "high": max(c["high"] for c in five_m),
            "low": min(c["low"] for c in five_m),
            "close": five_m[-1]["close"],
            "volume": sum(c["volume"] for c in five_m),
        }

        self._candles_1h[coin].append(h1)

        if self._cache:
            self._cache.upsert_candle("candles_1h", coin, h1)

        # 1시간봉 지표 계산 — 동일하게 스레드 풀 격리
        loop = asyncio.get_event_loop()
        df_1h = await loop.run_in_executor(None, self._build_df_with_indicators, coin, "1h")
        if self._on_candle_1h and df_1h is not None:
            await self._on_candle_1h(coin, df_1h)

    # ------------------------------------------------------------------
    # REST 갭 보완
    # ------------------------------------------------------------------

    async def _rest_gap_fill(self, coin: str, gap_ts: datetime) -> None:
        """WebSocket 갭 — REST 폴백으로 해당 캔들 1개만 보완."""
        if self._rest_fallback is None:
            return
        try:
            df = await self._rest_fallback(coin, "5m", gap_ts)
            if df is not None and not df.empty:
                row = df.iloc[-1].to_dict()
                row["coin"] = coin
                if "timestamp" not in row:
                    row["timestamp"] = gap_ts.isoformat()
                self._candles_5m[coin].append(row)
                if self._cache:
                    self._cache.upsert_candle("candles_5m", coin, row)
                logger.info("[%s] REST 갭 보완 완료 %s", coin, gap_ts)
        except Exception as exc:
            logger.error("[%s] REST 폴백 실패: %s", coin, exc)

    # ------------------------------------------------------------------
    # 일봉 업데이트 (매일 04:00 REST 1회, shift(1) 적용)
    # ------------------------------------------------------------------

    def daily_update(self, coin: str, df_daily_raw: pd.DataFrame) -> pd.DataFrame:
        """일봉 피처 생성 — Lookahead Bias 방지를 위해 shift(1) 필수 적용.

        Args:
            df_daily_raw: REST로 받은 원시 일봉 DataFrame (timestamp 인덱스)
        Returns:
            shift(1) 적용된 일봉 피처 DataFrame
        """
        df = df_daily_raw.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        df = df.sort_index()

        # 기술지표 계산
        close = df["close"].astype(float)
        df["ema50"] = ta.ema(close, length=50)
        df["ema200"] = ta.ema(close, length=200)
        df["rsi"] = ta.rsi(close, length=14)

        # 추세 인코딩 (골든크로스/데드크로스)
        ema50 = df["ema50"].fillna(0)
        ema200 = df["ema200"].fillna(0)
        df["trend_encoding"] = np.where(ema50 > ema200, 1, np.where(ema50 < ema200, -1, 0))

        # ★ Lookahead Bias 방지 — shift(1) 적용 (전일 확정값만 사용)
        shift_cols = ["ema50", "ema200", "rsi", "trend_encoding"]
        df_shifted = df.copy()
        df_shifted[shift_cols] = df[shift_cols].shift(1)

        # SQLite 저장
        if self._cache:
            for _, row in df_shifted.iterrows():
                candle_row = {
                    "coin": coin,
                    "timestamp": row.name.isoformat(),
                    "open": float(df.at[row.name, "open"]),
                    "high": float(df.at[row.name, "high"]),
                    "low": float(df.at[row.name, "low"]),
                    "close": float(df.at[row.name, "close"]),
                    "volume": float(df.at[row.name, "volume"]),
                    "ema50": float(row.get("ema50", 0) or 0),
                    "ema200": float(row.get("ema200", 0) or 0),
                    "rsi": float(row.get("rsi", 50) or 50),
                    "trend_encoding": int(row.get("trend_encoding", 0) or 0),
                }
                self._cache.upsert_candle("candles_1d", coin, candle_row)

        # 인메모리 캐시 갱신 — get_market_state_snapshot()이 일봉 피처를 읽기 위해.
        # shift(1) 결과의 마지막 non-NaN 행 사용 (첫 행만 NaN이고 나머지는 유효).
        valid = df_shifted[shift_cols].dropna()
        if not valid.empty:
            last = valid.iloc[-1]
            self._daily_features[coin] = {
                "ema50_1d":          float(last.get("ema50", 0) or 0),
                "ema200_1d":         float(last.get("ema200", 0) or 0),
                "rsi_1d":            float(last.get("rsi", 50) or 50),
                "trend_encoding_1d": int(last.get("trend_encoding", 0) or 0),
            }

        return df_shifted

    # ------------------------------------------------------------------
    # MarketState 스냅샷
    # ------------------------------------------------------------------

    def get_market_state_snapshot(self, coin: str) -> MarketState | None:
        """최신 인메모리 데이터 → MarketState (지표 최신값 추출)."""
        if len(self._candles_5m.get(coin, [])) < 14:
            return None

        df_5m = self._build_df_with_indicators(coin, "5m")
        if df_5m is None or df_5m.empty:
            return None

        latest = df_5m.iloc[-1]
        bucket = self._buckets.get(coin, _TradeBucket())
        now = datetime.now(timezone.utc)

        def _f(col: str, default: float = 0.0) -> float:
            val = latest.get(col, default)
            return float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else default

        # daily_update() shift(1) 결과 — 없으면 기본값 (daily_update 아직 미실행)
        daily = self._daily_features.get(coin, {})

        return MarketState(
            coin=coin,
            timestamp=now,
            open_5m=_f("open"),
            high_5m=_f("high"),
            low_5m=_f("low"),
            close_5m=_f("close"),
            volume_5m=_f("volume"),
            rsi_5m=_f("rsi", 50.0),
            macd_5m=_f("macd"),
            macd_signal_5m=_f("macd_signal"),
            bb_upper_5m=_f("bb_upper"),
            bb_lower_5m=_f("bb_lower"),
            ema7_5m=_f("ema7"),
            ema25_5m=_f("ema25"),
            ema99_5m=_f("ema99"),
            volume_ratio_5m=_f("volume_ratio", 1.0),
            adx_5m=_f("adx"),
            supertrend_signal=int(_f("supertrend_dir")),
            atr_5m=_f("atr"),
            tick_imbalance=bucket.tick_imbalance(),
            trade_velocity=bucket.trade_velocity(now),
            # 일봉 피처 (shift(1) 적용 완료) — Lookahead Bias 없음
            ema50_1d=daily.get("ema50_1d", 0.0),
            ema200_1d=daily.get("ema200_1d", 0.0),
            rsi_1d=daily.get("rsi_1d", 50.0),
            trend_encoding_1d=daily.get("trend_encoding_1d", 0),
        )

    # ------------------------------------------------------------------
    # 지표 계산
    # ------------------------------------------------------------------

    def _build_df_with_indicators(
        self, coin: str, tf: str
    ) -> pd.DataFrame | None:
        raw = list(self._candles_5m[coin] if tf == "5m" else self._candles_1h[coin])
        if len(raw) < 14:
            return None

        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return self._compute_indicators(df, tf)

    @staticmethod
    def _compute_indicators(df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """pandas-ta 지표 계산.

        일봉 지표는 daily_update()에서 별도 처리 (shift(1) 적용).
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI(14) — 공통
        rsi = ta.rsi(close, length=14)
        if rsi is not None:
            df["rsi"] = rsi

        # MACD(12,26,9) — 공통
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            cols = macd_df.columns.tolist()
            # pandas-ta 컬럼 순서: MACD, MACDh, MACDs
            df["macd"] = macd_df[cols[0]]
            df["macd_hist"] = macd_df[cols[1]]
            df["macd_signal"] = macd_df[cols[2]]

        if tf == "5m":
            # BB(20,2)
            bb = ta.bbands(close, length=20, std=2)
            if bb is not None and not bb.empty:
                bc = bb.columns.tolist()
                df["bb_lower"] = bb[bc[0]]
                df["bb_mid"] = bb[bc[1]]
                df["bb_upper"] = bb[bc[2]]

            # EMA(7, 25, 99)
            df["ema7"] = ta.ema(close, length=7)
            df["ema25"] = ta.ema(close, length=25)
            df["ema99"] = ta.ema(close, length=99)

            # ADX(14)
            adx_df = ta.adx(high, low, close, length=14)
            if adx_df is not None and not adx_df.empty:
                df["adx"] = adx_df.iloc[:, 0]

            # SuperTrend(10, 3)
            st = ta.supertrend(high, low, close, length=10, multiplier=3)
            if st is not None and not st.empty:
                st_cols = st.columns.tolist()
                df["supertrend"] = st[st_cols[0]]
                df["supertrend_dir"] = st[st_cols[1]]

            # ATR(14)
            df["atr"] = ta.atr(high, low, close, length=14)

            # 거래량변화율
            vol_ma = volume.rolling(20).mean()
            df["volume_ratio"] = volume / (vol_ma + 1e-9)

        elif tf == "1h":
            df["ema20"] = ta.ema(close, length=20)
            df["ema50"] = ta.ema(close, length=50)

            adx_df = ta.adx(high, low, close, length=14)
            if adx_df is not None and not adx_df.empty:
                df["adx"] = adx_df.iloc[:, 0]

            ema20 = df.get("ema20", pd.Series(dtype=float))
            ema50 = df.get("ema50", pd.Series(dtype=float))
            if not ema20.empty and not ema50.empty:
                df["trend_dir"] = np.where(ema20 > ema50, 1, -1)

        return df

    # ------------------------------------------------------------------
    # 유틸
    # ------------------------------------------------------------------

    @staticmethod
    def _floor_5m(dt: datetime) -> datetime:
        ts = dt.timestamp()
        return datetime.fromtimestamp(ts - ts % CANDLE_5M_SECONDS, tz=timezone.utc)

    def check_silent_drop(self, coin: str, threshold_sec: float = 300.0) -> bool:
        """마지막 trade 수신 후 threshold_sec 이상 침묵 → True."""
        last = self._last_trade_ts.get(coin)
        if last is None:
            return False
        return (datetime.now(timezone.utc) - last).total_seconds() > threshold_sec

    def get_current_price(self, coin: str) -> float | None:
        """현재 진행 중인 버킷의 최신 close 가격."""
        b = self._buckets.get(coin)
        if b is None or b.is_empty():
            return None
        return b.close_price

    def has_enough_data(self, coin: str, min_candles: int = 100) -> bool:
        """지표 계산 + ML 학습에 충분한 캔들 수 여부."""
        return len(self._candles_5m.get(coin, [])) >= min_candles
