"""
data/collector.py — 업비트 데이터 수집기 전체

포함 클래스:
  PairlistManager         — 동적 페어리스트 (매일 04:00 갱신, 상위 30개)
  RateLimiter             — REST 초당 8회 제한 (초기 로딩 시에만 적용)
  UpbitWebSocketCollector — WebSocket ticker + trade 수신
  MarketIndexCollector    — Fear&Greed, BTC 도미넌스, CoinGlass 펀딩비
  KimchiPremiumCollector  — 업비트 vs Binance BTC 가격 + 한국은행 환율
  OBICollector            — 업비트 오더북 불균형 (OBI, top5, wall_ratio)
  OnchainCollector        — CryptoQuant 거래소 유입/유출 (API 키 필요)
  SentimentCollector      — VADER + FinBERT ProcessPoolExecutor 분리
  UpbitDataCollector      — 전체 오케스트레이터 (초기 로딩 + 주기 갱신)
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Any

import ssl

import aiohttp
import certifi
import pyupbit

from schema import MarketState, RawMarketData

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

UPBIT_WS_URI = "wss://api.upbit.com/websocket/v1"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
BOK_EXCHANGE_RATE_URL = (
    "https://www.bok.or.kr/openapi/Stat/S_TOTAL_STATISTICS/statisticsSearch.json"
)
UPBIT_ORDERBOOK_URL = "https://api.upbit.com/v1/orderbook"
UPBIT_TICKER_URL = "https://api.upbit.com/v1/ticker"
UPBIT_MARKET_URL = "https://api.upbit.com/v1/market/all"

LEVERAGE_TOKEN_SUFFIXES = ("BULL", "BEAR", "UP", "DOWN")
LEVERAGE_TOKEN_SINGLE = ("L", "S")
MIN_VOLUME_KRW = 100_000_000     # 1억원
MAX_PAIRS = 30
WS_RECONNECT_MAX_DELAY = 60     # 지수 백오프 최대 60초


# ---------------------------------------------------------------------------
# RateLimiter — 초기 REST 로딩 전용
# ---------------------------------------------------------------------------

class RateLimiter:
    """초당 max_per_second 회 제한 (초기 히스토리 로딩 시만 사용)."""

    def __init__(self, max_per_second: int = 8) -> None:
        self._max = max_per_second
        self._timestamps: list[float] = []

    async def acquire(self) -> None:
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < 1.0]
        if len(self._timestamps) >= self._max:
            sleep_for = 1.0 - (now - self._timestamps[0])
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


# ---------------------------------------------------------------------------
# PairlistManager — 동적 페어리스트
# ---------------------------------------------------------------------------

class PairlistManager:
    """업비트 KRW 마켓 24h 거래량 상위 30개 동적 관리.

    - 매일 04:00 갱신 (APScheduler 호출)
    - 레버리지 토큰 / 블랙리스트 / 저유동성 자동 제외
    - 기존 보유 코인은 페어리스트 갱신 시 유지 (강제 청산 없음)
    """

    def __init__(
        self,
        blacklist: list[str] | None = None,
        cache=None,
    ) -> None:
        self._blacklist: set[str] = set(blacklist or [])
        self._cache = cache
        self._active_pairs: list[str] = []
        self._held_coins: set[str] = set()   # 현재 보유 포지션 코인

    async def refresh(self, session: aiohttp.ClientSession) -> list[str]:
        """거래량 상위 30개 갱신 + coin_scan_results 저장."""
        try:
            all_markets = await self._fetch_krw_markets(session)
            tickers = await self._fetch_tickers(session, all_markets)
            pairs = self._filter_pairs(tickers)
            self._active_pairs = pairs
            await self._save_scan_results(tickers, pairs)
            await self._save_coin_history_snapshot(pairs, tickers)
            logger.info("페어리스트 갱신 완료: %d개", len(pairs))
            return pairs
        except Exception as exc:
            logger.error("페어리스트 갱신 실패: %s", exc)
            if not self._active_pairs:
                # 첫 기동 시 이전 목록 없음 → 빈 목록 반환 시 WebSocket 구독 전무,
                # 데이터 수집 전면 중단. 즉시 종료해야 한다.
                raise RuntimeError(f"첫 기동 페어리스트 갱신 실패 — 봇을 시작할 수 없음: {exc}") from exc
            # 의도적 폴백: 갱신 실패 시 이전 목록 유지.
            # 빈 목록을 반환하면 WebSocket 구독이 취소되고 모든 데이터 수집이 멈추므로
            # 이전 페어리스트로 계속 운영하는 것이 안전하다.
            return self._active_pairs

    async def _fetch_krw_markets(self, session: aiohttp.ClientSession) -> list[str]:
        """업비트 KRW 마켓 전체 코인 목록."""
        async with session.get(UPBIT_MARKET_URL) as resp:
            data: list[dict] = await resp.json()
        return [m["market"] for m in data if m["market"].startswith("KRW-")]

    async def _fetch_tickers(
        self, session: aiohttp.ClientSession, markets: list[str]
    ) -> list[dict]:
        """최대 100개씩 분할하여 ticker 조회."""
        result: list[dict] = []
        chunk_size = 100
        for i in range(0, len(markets), chunk_size):
            chunk = markets[i : i + chunk_size]
            params = {"markets": ",".join(chunk)}
            async with session.get(UPBIT_TICKER_URL, params=params) as resp:
                result.extend(await resp.json())
            await asyncio.sleep(0.15)
        return result

    def _filter_pairs(self, tickers: list[dict]) -> list[str]:
        """제외 규칙 적용 후 상위 MAX_PAIRS개 반환."""
        filtered = []
        for t in tickers:
            market: str = t.get("market", "")
            coin = market.replace("KRW-", "")
            vol = float(t.get("acc_trade_price_24h", 0))

            # 저유동성
            if vol < MIN_VOLUME_KRW:
                continue
            # 블랙리스트
            if market in self._blacklist or coin in self._blacklist:
                continue
            # 레버리지 토큰
            if self._is_leverage_token(coin):
                continue

            filtered.append((market, vol))

        # 거래량 내림차순 정렬
        filtered.sort(key=lambda x: x[1], reverse=True)
        top_markets = [m for m, _ in filtered[:MAX_PAIRS]]

        # 보유 코인은 목록에 없어도 유지
        for coin in self._held_coins:
            if coin not in top_markets:
                top_markets.append(coin)

        return top_markets

    @staticmethod
    def _is_leverage_token(coin: str) -> bool:
        for suffix in LEVERAGE_TOKEN_SUFFIXES:
            if coin.endswith(suffix):
                return True
        for single in LEVERAGE_TOKEN_SINGLE:
            if coin.endswith(single) and len(coin) > 3:
                return True
        return False

    async def _save_scan_results(
        self, tickers: list[dict], included: list[str]
    ) -> None:
        if not self._cache:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for i, t in enumerate(tickers):
            market = t.get("market", "")
            rows.append({
                "timestamp": now,
                "coin": market,
                "rank_by_volume": i + 1,
                "volume_24h_krw": float(t.get("acc_trade_price_24h", 0)),
                "is_leverage_token": 1 if self._is_leverage_token(market.replace("KRW-", "")) else 0,
                "is_blacklisted": 1 if market in self._blacklist else 0,
                "included": 1 if market in included else 0,
                "reason_excluded": None if market in included else "volume_or_filter",
            })
        self._cache.bulk_upsert("coin_scan_results", rows)

    async def _save_coin_history_snapshot(
        self, pairs: list[str], tickers: list[dict]
    ) -> None:
        """생존 편향 처리용 일별 스냅샷 저장."""
        if not self._cache:
            return
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ticker_map = {t["market"]: t for t in tickers}
        rows = []
        for market in pairs:
            t = ticker_map.get(market, {})
            rows.append({
                "snapshot_date": today,
                "coin": market,
                "volume_24h_krw": float(t.get("acc_trade_price_24h", 0)),
                "rank": pairs.index(market) + 1,
                "market_cap_krw": None,
                "included_in_pairlist": 1,
            })
        self._cache.bulk_upsert("coin_history", rows)

    def get_active_pairs(self) -> list[str]:
        return list(self._active_pairs)

    def set_held_coins(self, coins: set[str]) -> None:
        self._held_coins = coins

    def is_active(self, market: str) -> bool:
        return market in self._active_pairs


# ---------------------------------------------------------------------------
# UpbitWebSocketCollector — WebSocket ticker + trade
# ---------------------------------------------------------------------------

class UpbitWebSocketCollector:
    """업비트 WebSocket 연결 관리.

    - 최대 30개 코인 단일 연결 동시 구독
    - ticker(현재가) + trade(체결) 수신 → asyncio.Queue
    - 연결 끊김 시 지수 백오프 자동 재연결
    - 페어리스트 갱신 시 재구독 자동 처리
    """

    def __init__(self, queue: asyncio.Queue[RawMarketData]) -> None:
        self._queue = queue
        self._pairs: list[str] = []
        self._running = False
        self._ws_task: asyncio.Task | None = None
        self._cb: Any | None = None  # CircuitBreaker — set_circuit_breaker()로 후(後) 주입

    def set_circuit_breaker(self, cb: Any) -> None:
        """CircuitBreaker 후(後) 주입."""
        self._cb = cb

    def set_pairs(self, pairs: list[str]) -> None:
        """페어리스트 갱신 시 호출 — 재연결 트리거."""
        self._pairs = pairs
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()

    async def start(self) -> None:
        self._running = True
        self._ws_task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()

    async def _run_forever(self) -> None:
        delay = 1.0
        while self._running:
            try:
                await self._connect_and_receive()
                delay = 1.0  # 정상 종료 시 초기화
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("WebSocket 연결 끊김: %s — %.1f초 후 재연결", exc, delay)
                if self._cb is not None:
                    self._cb.record_api_error()
                await asyncio.sleep(delay)
                delay = min(delay * 2, WS_RECONNECT_MAX_DELAY)

    async def _connect_and_receive(self) -> None:
        if not self._pairs:
            await asyncio.sleep(5)
            return

        subscribe_msg = self._build_subscribe(self._pairs)
        headers = {"User-Agent": "Mozilla/5.0"}  # Origin 제거 → rate limit 완화

        try:
            import websockets  # type: ignore
        except ImportError:
            logger.error("websockets 패키지 미설치 — pip install websockets")
            await asyncio.sleep(30)
            return

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with websockets.connect(
            UPBIT_WS_URI, additional_headers=headers, ssl=ssl_ctx
        ) as ws:
            await ws.send(json.dumps(subscribe_msg))
            logger.info("WebSocket 구독 시작: %d코인", len(self._pairs))

            async for raw in ws:
                if not self._running:
                    break
                data = json.loads(raw)
                msg = self._parse_message(data)
                if msg:
                    await self._queue.put(msg)

    @staticmethod
    def _build_subscribe(pairs: list[str]) -> list[dict]:
        ticket = [{"ticket": str(uuid.uuid4())}]
        ticker_type = [{"type": "ticker", "codes": pairs, "isOnlyRealtime": True}]
        trade_type = [{"type": "trade", "codes": pairs, "isOnlyRealtime": True}]
        return ticket + ticker_type + trade_type

    @staticmethod
    def _parse_message(data: dict) -> RawMarketData | None:
        try:
            t = data.get("type")
            market = data.get("code", "")
            if not market.startswith("KRW-"):
                return None

            ts_ms = data.get("trade_timestamp") or data.get("timestamp", 0)
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

            return RawMarketData(
                coin=market,
                timestamp=ts,
                trade_price=float(data.get("trade_price", 0)),
                trade_volume=float(data.get("trade_volume", 0)),
                ask_bid=data.get("ask_bid", "ASK"),
                sequential_id=int(data.get("sequential_id", 0)),
                stream_type=t or "unknown",
                acc_trade_volume_24h=data.get("acc_trade_volume_24h"),
                acc_trade_price_24h=data.get("acc_trade_price_24h"),
                change_rate=data.get("change_rate"),
            )
        except Exception as exc:
            logger.warning("WebSocket 메시지 파싱 오류: %s", exc)
            # 의도적 계속 진행: 단일 메시지 파싱 실패는 스트림 전체 중단 사유 아님
            return None


# ---------------------------------------------------------------------------
# MarketIndexCollector — Fear&Greed, BTC 도미넌스, 펀딩비
# ---------------------------------------------------------------------------

class MarketIndexCollector:
    """시장 지수 수집 (Alternative.me, CoinGecko, CoinGlass)."""

    def __init__(self, cache=None) -> None:
        self._cache = cache
        self._fear_greed: float = 50.0
        self._btc_dominance: float = 50.0
        self._altcoin_season: float = 50.0
        self._funding_rate: float = 0.0

    async def fetch_fear_greed(self, session: aiohttp.ClientSession) -> float:
        """Alternative.me Fear & Greed — 일 1회."""
        try:
            async with session.get(FEAR_GREED_URL, timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json(content_type=None)
            value = float(data["data"][0]["value"])
            self._fear_greed = value
            self._save_index()
            return value
        except Exception as exc:
            logger.warning("Fear&Greed 수집 실패: %s", exc)
            return self._fear_greed

    async def fetch_btc_dominance(self, session: aiohttp.ClientSession) -> float:
        """CoinGecko BTC 도미넌스 — 1시간."""
        try:
            async with session.get(
                COINGECKO_GLOBAL_URL, timeout=aiohttp.ClientTimeout(total=15)
            ) as r:
                data = await r.json(content_type=None)
            dom = float(data["data"]["market_cap_percentage"]["btc"])
            self._btc_dominance = dom
            self._save_index()
            return dom
        except Exception as exc:
            logger.warning("BTC 도미넌스 수집 실패: %s", exc)
            return self._btc_dominance

    def _save_index(self) -> None:
        if not self._cache:
            return
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fear_greed": self._fear_greed,
            "btc_dominance": self._btc_dominance,
            "btc_market_cap_usd": None,
            "altcoin_season": self._altcoin_season,
            "funding_rate": self._funding_rate,
            "usd_krw_rate": None,
        }
        self._cache.upsert_candle("market_indices", "", row)

    @property
    def fear_greed(self) -> float:
        return self._fear_greed

    @property
    def btc_dominance(self) -> float:
        return self._btc_dominance


# ---------------------------------------------------------------------------
# KimchiPremiumCollector — 김치프리미엄 (5분 주기)
# ---------------------------------------------------------------------------

class KimchiPremiumCollector:
    """업비트 BTC/KRW vs Binance BTC/USDT × USD/KRW 환율.

    kimchi_premium = (upbit_krw / (binance_usd × usd_krw) - 1) × 100
    환율: 한국은행 OpenAPI (일 1회 캐시)
    """

    def __init__(self, bok_api_key: str = "", cache=None, usd_krw_initial: float = 1350.0) -> None:
        self._bok_api_key = bok_api_key
        self._cache = cache
        self._usd_krw: float = usd_krw_initial  # BOK API 실패 시 fallback — config.yaml data.usd_krw_initial
        self._last_bok_date: str = ""
        self._latest_premium: float = 0.0
        self._candle_builder: Any = None  # set_candle_builder()로 주입 — BTC 현재가 WebSocket 우선

    def set_candle_builder(self, candle_builder: Any) -> None:
        """CandleBuilder 주입 — BTC/KRW 현재가를 WebSocket 데이터에서 읽기 위해."""
        self._candle_builder = candle_builder

    async def refresh_usd_krw(self, session: aiohttp.ClientSession) -> float:
        """한국은행 환율 (일 1회 캐시)."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if self._last_bok_date == today:
            return self._usd_krw

        if not self._bok_api_key:
            logger.warning("BOK API 키 없음 — USD/KRW 환율 캐시값 사용: %.1f (김치프리미엄 정확도 저하)", self._usd_krw)
            return self._usd_krw

        try:
            params = {
                "auth": self._bok_api_key,
                "lang": "kr",
                "startTime": today,
                "endTime": today,
                "stat_code": "S_TOTAL_STATISTICS",
                "item_code1": "0000001",
            }
            async with session.get(
                BOK_EXCHANGE_RATE_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as r:
                data = await r.json(content_type=None)
            rate = float(data["StatisticSearch"]["row"][0]["DATA_VALUE"])
            self._usd_krw = rate
            self._last_bok_date = today
            logger.debug("USD/KRW 환율 갱신: %.2f", rate)
        except Exception as exc:
            logger.warning("BOK 환율 API 실패: %s — 캐시값 사용", exc)

        return self._usd_krw

    async def collect(self, session: aiohttp.ClientSession) -> float:
        """5분 주기 김치프리미엄 계산 + SQLite 저장."""
        try:
            # 업비트 BTC/KRW — WebSocket CandleBuilder 우선, 미수신 시 run_in_executor 격리
            # pyupbit.get_current_price()는 requests 기반 동기 라이브러리 — 직접 호출 금지
            upbit_krw_val: float | None = None
            if self._candle_builder is not None:
                upbit_krw_val = self._candle_builder.get_current_price("KRW-BTC")
            if upbit_krw_val is None:
                # WebSocket 데이터 미수신 또는 CandleBuilder 미주입 — 동기 호출을 스레드 풀로 격리
                loop = asyncio.get_event_loop()
                upbit_krw_val = await loop.run_in_executor(
                    None, pyupbit.get_current_price, "KRW-BTC"
                )
            if not upbit_krw_val:
                logger.warning("업비트 BTC/KRW 조회 실패 — 김치프리미엄 이전값 유지: %.2f%%", self._latest_premium)
                return self._latest_premium
            upbit_krw = float(upbit_krw_val)

            # 바이낸스 BTC/USDT
            async with session.get(
                BINANCE_TICKER_URL, timeout=aiohttp.ClientTimeout(total=5)
            ) as r:
                bdata = await r.json(content_type=None)
            binance_usd = float(bdata["price"])

            # 환율
            usd_krw = await self.refresh_usd_krw(session)

            premium = (upbit_krw / (binance_usd * usd_krw) - 1) * 100
            self._latest_premium = premium

            if self._cache:
                self._cache.upsert_candle(
                    "kimchi_premium_log",
                    "",
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "upbit_btc_krw": upbit_krw,
                        "binance_btc_usd": binance_usd,
                        "usd_krw_rate": usd_krw,
                        "kimchi_premium_pct": premium,
                    },
                )
            logger.debug("김치프리미엄: %.2f%%", premium)
            return premium

        except Exception as exc:
            logger.error("김치프리미엄 수집 실패: %s — 이전값 유지 (의도적 폴백: 단기 API 오류 시 캐시 사용)", exc)
            return self._latest_premium

    @property
    def latest_premium(self) -> float:
        return self._latest_premium


# ---------------------------------------------------------------------------
# OBICollector — 오더북 불균형
# ---------------------------------------------------------------------------

class OBICollector:
    """업비트 REST /v1/orderbook — OBI + top5_concentration + wall_ratio.

    [주의] 오더북 원본 저장 금지 — 계산된 float 값만 MarketState에 통합.
    """

    def __init__(self) -> None:
        self._latest: dict[str, dict[str, float]] = {}

    async def collect(
        self, session: aiohttp.ClientSession, pairs: list[str]
    ) -> dict[str, dict[str, float]]:
        """30개 코인 오더북 OBI 수집.

        Returns:
            {coin: {obi, top5_concentration, orderbook_wall_ratio, latency_ms}}
        """
        result: dict[str, dict[str, float]] = {}
        markets_param = ",".join(pairs[:MAX_PAIRS])
        t0 = time.monotonic()
        try:
            async with session.get(
                UPBIT_ORDERBOOK_URL,
                params={"markets": markets_param},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                data: list[dict] = await r.json(content_type=None)
            latency_ms = (time.monotonic() - t0) * 1000

            for book in data:
                market = book.get("market", "")
                units: list[dict] = book.get("orderbook_units", [])
                if not units:
                    continue

                bid_total = sum(float(u.get("bid_size", 0)) for u in units)
                ask_total = sum(float(u.get("ask_size", 0)) for u in units)
                total = bid_total + ask_total

                obi = (bid_total - ask_total) / (total + 1e-9)

                top5_bid = sum(float(u.get("bid_size", 0)) for u in units[:5])
                top5_ask = sum(float(u.get("ask_size", 0)) for u in units[:5])
                top5_total = top5_bid + top5_ask
                top5_concentration = top5_total / (total + 1e-9)

                all_sizes = [float(u.get("bid_size", 0)) for u in units] + \
                            [float(u.get("ask_size", 0)) for u in units]
                max_size = max(all_sizes) if all_sizes else 0
                avg_size = sum(all_sizes) / (len(all_sizes) + 1e-9)
                wall_ratio = max_size / (avg_size + 1e-9)

                result[market] = {
                    "obi": round(obi, 4),
                    "top5_concentration": round(top5_concentration, 4),
                    "orderbook_wall_ratio": round(wall_ratio, 2),
                    "latency_ms": round(latency_ms, 1),
                }

        except Exception as exc:
            logger.warning("OBI 수집 실패: %s", exc)

        self._latest.update(result)
        return result

    def get(self, market: str) -> dict[str, float]:
        return self._latest.get(market, {
            "obi": 0.0,
            "top5_concentration": 0.5,
            "orderbook_wall_ratio": 1.0,
            "latency_ms": 0.0,
        })


# ---------------------------------------------------------------------------
# OnchainCollector — CryptoQuant 거래소 유입/유출 (API 키 필요)
# ---------------------------------------------------------------------------

class OnchainCollector:
    """CryptoQuant 무료 티어 — 거래소 유입/유출량 1시간 주기.

    API 키 없으면 더미값 반환 (시스템 동작 보장).
    """

    def __init__(self, api_key: str = "", cache=None) -> None:
        self._api_key = api_key
        self._cache = cache
        self._latest: dict[str, dict[str, float]] = {}

    async def collect(
        self, session: aiohttp.ClientSession, coins: list[str]
    ) -> dict[str, dict[str, float]]:
        if not self._api_key:
            logger.debug("CryptoQuant API 키 없음 — 온체인 수집 스킵")
            return {c: {"exchange_inflow": 0.0, "exchange_outflow": 0.0, "net_flow": 0.0}
                    for c in coins}

        result: dict[str, dict[str, float]] = {}
        for coin in coins:
            symbol = coin.replace("KRW-", "").lower()
            try:
                url = f"https://api.cryptoquant.com/v1/btc/exchange-flows/inflow?exchange=all&window=hour&limit=1"
                headers = {"Authorization": f"Bearer {self._api_key}"}
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    data = await r.json(content_type=None)
                inflow = float(data.get("result", {}).get("data", [{}])[0].get("inflow_total", 0))
                outflow = float(data.get("result", {}).get("data", [{}])[0].get("outflow_total", 0))
                result[coin] = {
                    "exchange_inflow": inflow,
                    "exchange_outflow": outflow,
                    "net_flow": outflow - inflow,
                }
                if self._cache:
                    self._cache.upsert_candle("onchain_data", coin, {
                        "coin": coin,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        **result[coin],
                    })
            except Exception as exc:
                logger.debug("온체인 %s 수집 실패: %s", coin, exc)
                result[coin] = {"exchange_inflow": 0.0, "exchange_outflow": 0.0, "net_flow": 0.0}

        self._latest.update(result)
        return result

    def get(self, coin: str) -> dict[str, float]:
        return self._latest.get(coin, {
            "exchange_inflow": 0.0,
            "exchange_outflow": 0.0,
            "net_flow": 0.0,
        })


# ---------------------------------------------------------------------------
# FinBERT 분리 실행 함수 (ProcessPoolExecutor 전용 — 메인 루프와 격리)
# ---------------------------------------------------------------------------

def _finbert_batch_worker(texts: list[str]) -> list[float]:
    """별도 프로세스에서 실행 — asyncio 블로킹 방지.

    반드시 ProcessPoolExecutor(max_workers=1)에서만 호출.
    완료 후 del model; gc.collect() → 메모리 즉시 해제.
    """
    import gc  # noqa
    scores: list[float] = []
    model = None
    try:
        from transformers import pipeline  # type: ignore
        model = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=-1,   # CPU 강제 (MPS 사용 금지)
            truncation=True,
            max_length=512,
        )
        for text in texts:
            out = model(text[:512])[0]
            label = out["label"].lower()
            score = out["score"]
            if label == "positive":
                scores.append(score)
            elif label == "negative":
                scores.append(-score)
            else:
                scores.append(0.0)
    except Exception as e:
        scores = [0.0] * len(texts)
        # ProcessPoolExecutor 격리 프로세스이므로 logger 대신 직접 출력 후,
        # 부모 프로세스의 SentimentCollector가 FinBERT 실패를 감지하고 VADER로 폴백한다.
        import logging as _logging
        _logging.getLogger(__name__).error("[FinBERT Worker] 오류: %s — 더미 점수(0.0) 반환", e)
    finally:
        if model is not None:
            del model
        gc.collect()
    return scores


# ---------------------------------------------------------------------------
# SentimentCollector — VADER + FinBERT
# ---------------------------------------------------------------------------

class SentimentCollector:
    """RSS 뉴스 감성 분석.

    1단계: VADER 전체 분석 (compound ±0.3 이상 즉시 확정)
    2단계: ±0.3 미만만 FinBERT 2차 분석 (ProcessPoolExecutor 격리)
    폴백: FinBERT 오류 시 VADER 단독 결과 사용.
    """

    RSS_URLS = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]

    def __init__(
        self,
        process_pool: ProcessPoolExecutor,
        cache=None,
    ) -> None:
        self._pool = process_pool
        self._cache = cache
        self._latest_score: float = 0.0
        self._latest_confidence: float = 0.0
        self._vader = None

    def _get_vader(self):
        if self._vader is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
                self._vader = SentimentIntensityAnalyzer()
            except ImportError:
                logger.warning("vaderSentiment 미설치 — pip install vaderSentiment")
        return self._vader

    async def collect(
        self,
        session: aiohttp.ClientSession,
        coin: str = "BTC",
    ) -> tuple[float, float]:
        """뉴스 수집 + 감성 분석 → (score, confidence).

        score: -1~+1, confidence: 0~1
        """
        texts = await self._fetch_news(session)
        if not texts:
            return self._latest_score, self._latest_confidence

        vader = self._get_vader()
        vader_scores: list[float] = []
        uncertain_indices: list[int] = []
        uncertain_texts: list[str] = []

        for i, text in enumerate(texts):
            if vader:
                compound = vader.polarity_scores(text)["compound"]
            else:
                compound = 0.0
            vader_scores.append(compound)
            if abs(compound) < 0.3:
                uncertain_indices.append(i)
                uncertain_texts.append(text)

        # FinBERT: 불확실 케이스만 별도 프로세스
        finbert_map: dict[int, float] = {}
        if uncertain_texts:
            try:
                loop = asyncio.get_event_loop()
                fb_scores: list[float] = await loop.run_in_executor(
                    self._pool, _finbert_batch_worker, uncertain_texts
                )
                for idx, fb_score in zip(uncertain_indices, fb_scores):
                    finbert_map[idx] = fb_score
            except Exception as exc:
                logger.warning("FinBERT 실패 — VADER 단독 사용: %s", exc)

        # 최종 점수 병합
        final_scores: list[float] = []
        for i, vs in enumerate(vader_scores):
            if i in finbert_map:
                final_scores.append(finbert_map[i])
            else:
                final_scores.append(vs)

        if not final_scores:
            return self._latest_score, self._latest_confidence

        avg_score = sum(final_scores) / len(final_scores)
        confidence = min(1.0, abs(avg_score) * 2)

        self._latest_score = avg_score
        self._latest_confidence = confidence

        if self._cache:
            self._cache.upsert_candle("sentiment_log", coin, {
                "coin": coin,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "vader_score": sum(vader_scores) / len(vader_scores),
                "finbert_score": sum(finbert_map.values()) / max(len(finbert_map), 1),
                "combined_score": avg_score,
                "confidence": confidence,
                "news_count": len(texts),
                "veto_triggered": 0,
            })

        return avg_score, confidence

    async def _fetch_news(self, session: aiohttp.ClientSession) -> list[str]:
        texts: list[str] = []
        for url in self.RSS_URLS:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    raw = await r.text()
                # 간단한 title 추출 (feedparser 없이)
                import re
                titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", raw)
                if not titles:
                    titles = re.findall(r"<title>(.*?)</title>", raw)
                texts.extend(titles[:20])
            except Exception as exc:
                logger.debug("RSS 수집 실패 %s: %s", url, exc)
        return texts

    @property
    def latest_score(self) -> float:
        return self._latest_score

    @property
    def latest_confidence(self) -> float:
        return self._latest_confidence


# ---------------------------------------------------------------------------
# UpbitDataCollector — 전체 오케스트레이터
# ---------------------------------------------------------------------------

class UpbitDataCollector:
    """데이터 수집 전체 오케스트레이터.

    초기화 순서:
      1. init_db()
      2. initial_history_load() — REST 1회 (코인당 200개)
      3. start_websocket() — 이후 실시간 WebSocket 합성으로 전환

    주기 수집 (APScheduler 호출):
      collect_market_indices()  — 1시간
      collect_kimchi_premium()  — 5분
      collect_obi()             — 5분
      collect_sentiment()       — 1시간
      collect_onchain()         — 1시간
      refresh_pairlist()        — 매일 04:00
    """

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        db_path: str,
        bok_api_key: str = "",
        cryptoquant_key: str = "",
        blacklist: list[str] | None = None,
        finbert_pool: ProcessPoolExecutor | None = None,
        yaml_config: dict | None = None,
    ) -> None:
        from data.cache import CandleCache, init_db

        init_db(db_path)
        self._cache = CandleCache(db_path)
        self._access_key = access_key
        self._secret_key = secret_key

        self._pairlist = PairlistManager(blacklist=blacklist, cache=self._cache)
        self._ws_queue: asyncio.Queue[RawMarketData] = asyncio.Queue(maxsize=10_000)
        self._ws_collector = UpbitWebSocketCollector(self._ws_queue)
        self._market_idx = MarketIndexCollector(cache=self._cache)
        _usd_krw_init = float((yaml_config or {}).get("data", {}).get("usd_krw_initial", 1350.0))
        self._kimchi = KimchiPremiumCollector(
            bok_api_key=bok_api_key, cache=self._cache, usd_krw_initial=_usd_krw_init
        )
        self._obi = OBICollector()
        self._onchain = OnchainCollector(api_key=cryptoquant_key, cache=self._cache)

        _pool = finbert_pool or ProcessPoolExecutor(max_workers=1)
        self._sentiment = SentimentCollector(process_pool=_pool, cache=self._cache)

        from data.candle_builder import CandleBuilder
        self._candle_builder = CandleBuilder(cache=self._cache)
        self._kimchi.set_candle_builder(self._candle_builder)

        self._rate_limiter = RateLimiter(max_per_second=8)
        self._session: aiohttp.ClientSession | None = None
        self._cb: Any | None = None  # CircuitBreaker — set_circuit_breaker()로 후(後) 주입
        self._ws_queue_task: asyncio.Task | None = None  # process_ws_queue 태스크 핸들

    def set_circuit_breaker(self, cb: Any) -> None:
        """CircuitBreaker 후(後) 주입 — engine 초기화 후 main.py에서 호출."""
        self._cb = cb
        self._ws_collector.set_circuit_breaker(cb)

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """aiohttp 세션 생성 + 초기 로딩 + WebSocket 시작."""
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        self._session = aiohttp.ClientSession(connector=connector)
        await self._refresh_pairlist()
        await self._initial_history_load()
        await self._ws_collector.start()
        # ws_queue → CandleBuilder 소비 태스크 시작 (stop()에서 취소)
        self._ws_queue_task = asyncio.create_task(
            self.process_ws_queue(), name="ws_queue_consumer"
        )
        logger.info("UpbitDataCollector 시작 완료")

    async def stop(self) -> None:
        if self._ws_queue_task and not self._ws_queue_task.done():
            self._ws_queue_task.cancel()
            await asyncio.gather(self._ws_queue_task, return_exceptions=True)
        await self._ws_collector.stop()
        if self._session:
            await self._session.close()
        self._cache.close()

    # ------------------------------------------------------------------
    # 초기 히스토리 로딩 (REST 1회성)
    # ------------------------------------------------------------------

    async def _initial_history_load(self) -> None:
        """코인당 200개 캔들 REST 로딩 — 최초 기동 시 1회만."""
        pairs = self._pairlist.get_active_pairs()
        logger.info("초기 히스토리 로딩 시작: %d코인", len(pairs))

        for coin in pairs:
            # 이미 데이터 있으면 스킵
            latest = self._cache.get_latest_timestamp("candles_5m", coin)
            if latest is not None:
                logger.debug("[%s] 기존 데이터 존재 — 초기 로딩 스킵", coin)
                continue

            await self._rate_limiter.acquire()
            try:
                df = pyupbit.get_ohlcv(coin, interval="minute5", count=200)
                if df is not None and not df.empty:
                    rows = self._df_to_rows(coin, df, "5m")
                    self._cache.bulk_upsert("candles_5m", rows)

                    # 인메모리 CandleBuilder에도 로드
                    for row in rows:
                        self._candle_builder.add_historical_candle(coin, row)

                await asyncio.sleep(0.1)
            except Exception as exc:
                logger.error("[%s] 초기 5분봉 로딩 실패: %s", coin, exc)

            # 1시간봉
            await self._rate_limiter.acquire()
            try:
                df_1h = pyupbit.get_ohlcv(coin, interval="minute60", count=200)
                if df_1h is not None and not df_1h.empty:
                    rows_1h = self._df_to_rows(coin, df_1h, "1h")
                    self._cache.bulk_upsert("candles_1h", rows_1h)
                await asyncio.sleep(0.1)
            except Exception as exc:
                logger.error("[%s] 초기 1시간봉 로딩 실패: %s", coin, exc)

        logger.info("초기 히스토리 로딩 완료")

    async def daily_candle_update(self) -> None:
        """일봉 REST 갱신 — 매일 04:00 1회 (30코인 × 1요청 = 30회)."""
        pairs = self._pairlist.get_active_pairs()
        for coin in pairs:
            await self._rate_limiter.acquire()
            try:
                df = pyupbit.get_ohlcv(coin, interval="day", count=400)
                if df is not None and not df.empty:
                    df.index = pd.to_datetime(df.index, utc=True) if hasattr(df.index, 'tz') else df.index
                    # daily_update()는 shift(1) 적용 후 SQLite(candles_1d)에 저장.
                    # 반환값(df_shifted)은 현재 사용 안 함 — get_market_state_snapshot()이
                    # 일봉 피처를 SQLite에서 직접 읽는 경로가 구현되면 여기서 활용 예정.
                    self._candle_builder.daily_update(coin, df)
                await asyncio.sleep(0.1)
            except Exception as exc:
                logger.error("[%s] 일봉 갱신 실패: %s", coin, exc)

    # ------------------------------------------------------------------
    # 주기 수집
    # ------------------------------------------------------------------

    async def _refresh_pairlist(self) -> None:
        assert self._session
        pairs = await self._pairlist.refresh(self._session)
        self._ws_collector.set_pairs(pairs)

    async def collect_market_indices(self) -> None:
        assert self._session
        await asyncio.gather(
            self._market_idx.fetch_fear_greed(self._session),
            self._market_idx.fetch_btc_dominance(self._session),
            return_exceptions=True,
        )

    async def collect_kimchi_premium(self) -> float:
        assert self._session
        return await self._kimchi.collect(self._session)

    async def collect_obi(self) -> dict[str, dict[str, float]]:
        assert self._session
        return await self._obi.collect(self._session, self._pairlist.get_active_pairs())

    async def collect_sentiment(self) -> tuple[float, float]:
        assert self._session
        return await self._sentiment.collect(self._session)

    async def collect_onchain(self) -> None:
        assert self._session
        await self._onchain.collect(self._session, self._pairlist.get_active_pairs())

    # ------------------------------------------------------------------
    # WebSocket 큐 소비 → CandleBuilder
    # ------------------------------------------------------------------

    async def process_ws_queue(self) -> None:
        """asyncio.Queue에서 RawMarketData 꺼내 CandleBuilder 전달.

        stop() 호출 시 태스크가 cancel되어 CancelledError로 종료.
        on_trade() 예외는 로깅 후 루프 유지 (서킷브레이커 오류 기록).
        """
        while True:
            try:
                msg = await self._ws_queue.get()
            except asyncio.CancelledError:
                break  # stop() → task.cancel() 시 정상 종료
            try:
                await self._candle_builder.on_trade(msg)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[Collector] CandleBuilder.on_trade() 실패: %s", exc)
                if self._cb is not None:
                    self._cb.record_api_error()
                # 예외 발생 시 루프 유지 — 다음 메시지 처리 계속

    # ------------------------------------------------------------------
    # MarketState 조회
    # ------------------------------------------------------------------

    async def get_market_state(self, coin: str) -> MarketState | None:
        """최신 MarketState 반환 — 외부 피처(지수/감성/온체인/OBI)도 통합.

        get_market_state_snapshot() 내부에서 pandas-ta 지표 계산(_build_df_with_indicators)이
        실행되므로 run_in_executor로 격리해 이벤트 루프 블로킹을 방지한다.
        candle_builder.get_market_state_snapshot()은 sync def 그대로 유지.
        """
        loop = asyncio.get_event_loop()
        ms = await loop.run_in_executor(
            None, self._candle_builder.get_market_state_snapshot, coin
        )
        if ms is None:
            return None

        # 시장 지수
        ms.fear_greed = self._market_idx.fear_greed
        ms.btc_dominance = self._market_idx.btc_dominance

        # 감성
        ms.sentiment_score = self._sentiment.latest_score
        ms.sentiment_confidence = self._sentiment.latest_confidence

        # 김치프리미엄
        ms.kimchi_premium = self._kimchi.latest_premium

        # OBI
        obi_data = self._obi.get(coin)
        ms.obi = obi_data.get("obi", 0.0)
        ms.top5_concentration = obi_data.get("top5_concentration", 0.5)
        ms.orderbook_wall_ratio = obi_data.get("orderbook_wall_ratio", 1.0)
        ms.api_latency_ms = obi_data.get("latency_ms", 0.0)

        # 온체인
        oc = self._onchain.get(coin)
        ms.exchange_inflow = oc.get("exchange_inflow", 0.0)
        ms.exchange_outflow = oc.get("exchange_outflow", 0.0)

        return ms

    # ------------------------------------------------------------------
    # 유틸
    # ------------------------------------------------------------------

    @staticmethod
    def _df_to_rows(coin: str, df: Any, tf: str) -> list[dict]:
        """pyupbit DataFrame → SQLite row 리스트."""
        import pandas as pd  # noqa
        rows = []
        for ts, row in df.iterrows():
            ts_str = pd.Timestamp(ts, tz="UTC").isoformat() if not hasattr(ts, "isoformat") else ts.isoformat()
            rows.append({
                "coin": coin,
                "timestamp": ts_str,
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
            })
        return rows

    @property
    def candle_builder(self):
        return self._candle_builder

    @property
    def pairlist(self) -> PairlistManager:
        return self._pairlist

    @property
    def ws_queue(self) -> asyncio.Queue:
        return self._ws_queue

    async def health_check(self) -> dict[str, Any]:
        """헬스체크 — WebSocket 상태 + 데이터 신선도 보고."""
        pairs = self._pairlist.get_active_pairs()
        stale_coins = [
            c for c in pairs
            if self._candle_builder.check_silent_drop(c, threshold_sec=600)
        ]
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_pairs": len(pairs),
            "stale_coins": stale_coins,
            "ws_queue_size": self._ws_queue.qsize(),
            "fear_greed": self._market_idx.fear_greed,
            "btc_dominance": self._market_idx.btc_dominance,
            "kimchi_premium": self._kimchi.latest_premium,
        }


# pandas import (daily_candle_update 내부 사용)
try:
    import pandas as pd
except ImportError:
    pass
