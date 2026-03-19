"""
layers/layer1_filter.py — Layer 1 룰 기반 시장 필터

10개 조건을 순서대로 체크 (단락 평가).
하나라도 BLOCK이면 tradeable=False 즉시 반환.

조건 0: 서킷브레이커 상태 (최우선)
조건 1: ADX 신뢰도 (Phase A/B: ADX 단독, <15 → 보류)
조건 2: Fear&Greed 15~85 범위
조건 3: BTC 도미넌스 < 60% (초과 시 ×0.7)
조건 4: 일봉 EMA50 vs EMA200 (데드크로스 ×0.8)
조건 5: 멀티타임프레임 추세 합의 (5m/1h/1d)
조건 6: 5분봉 거래량 20기간 평균 50% 이상
조건 7: 온체인 거래소 유입 전일比 +50% 경고
조건 8: 감성 비토 (레짐 연동)
조건 9: 코인 클러스터 중복 포지션 (Phase C, 현재 스킵)
조건 10: API 레이턴시 < 500ms
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from risk.circuit_breaker import CircuitBreaker
from schema import FilterResult, MarketState

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

ADX_STRONG_THRESHOLD = 30
ADX_NORMAL_THRESHOLD = 20
ADX_MIN_THRESHOLD = 15          # 미만 시 매수 보류
FEAR_GREED_MIN = 15
FEAR_GREED_MAX = 85
BTC_DOM_THRESHOLD = 60.0        # % 초과 시 알트 신호 ×0.7
SENTIMENT_VETO_THRESHOLD = -0.3  # 부정 감성 비토 기준
LATENCY_MAX_MS = 500.0
KIMCHI_PREMIUM_HIGH = 5.0       # % 초과 시 과열 경고 ×0.8
KIMCHI_PREMIUM_LOW = -2.0       # % 미만 시 역프리미엄 보류
VOLUME_RATIO_MIN = 0.5          # 20기간 평균 대비 50%
ONCHAIN_INFLOW_SURGE = 1.5      # 전일 대비 +50% = 1.5배


class Layer1MarketFilter:
    """Layer 1 룰 기반 시장 필터.

    사용법:
        cb = CircuitBreaker()
        filt = Layer1MarketFilter(circuit_breaker=cb, cache=cache)
        result = await filt.check(market_state, coin="KRW-BTC")
    """

    def __init__(
        self,
        circuit_breaker: CircuitBreaker,
        cache=None,
        phase_c_enabled: bool = False,      # HMM / 클러스터 조건 활성화 (Phase C)
        prev_onchain: dict[str, float] | None = None,  # 전일 온체인 유입량 캐시
    ) -> None:
        self._cb = circuit_breaker
        self._cache = cache
        self._phase_c = phase_c_enabled
        self._prev_onchain: dict[str, float] = prev_onchain or {}

    # ------------------------------------------------------------------
    # 메인 진입점
    # ------------------------------------------------------------------

    async def check(self, ms: MarketState, coin: str) -> FilterResult:
        """MarketState → FilterResult (10개 조건 순차 체크)."""
        warnings: list[str] = []
        multiplier = 1.0
        tradeable = True

        # ------ 조건 0: 서킷브레이커 ------
        cb_ok, cb_level, cb_warn = self._cond0_circuit_breaker()
        if cb_warn:
            warnings.append(cb_warn)
        if not cb_ok:
            return self._build_result(
                ms, coin, tradeable=False,
                multiplier=0.0, warnings=warnings,
                cb_level=cb_level,
            )

        # ------ 조건 1: ADX 신뢰도 ------
        adx_ok, adx_warn, regime = self._cond1_adx_regime(ms)
        if adx_warn:
            warnings.append(adx_warn)
        if not adx_ok:
            return self._build_result(
                ms, coin, tradeable=False,
                multiplier=0.0, warnings=warnings,
                regime_strategy=regime,
            )

        # ------ 조건 2: Fear&Greed ------
        fg_ok, fg_warn = self._cond2_fear_greed(ms)
        if fg_warn:
            warnings.append(fg_warn)
        if not fg_ok:
            return self._build_result(
                ms, coin, tradeable=False,
                multiplier=0.0, warnings=warnings,
                regime_strategy=regime,
            )

        # ------ 조건 3: BTC 도미넌스 ------
        dom_ok, dom_mult, dom_warn = self._cond3_btc_dominance(ms)
        if dom_warn:
            warnings.append(dom_warn)
        multiplier *= dom_mult
        # 도미넌스 초과는 차단이 아닌 신호 감소

        # ------ 조건 4: 일봉 EMA 크로스 ------
        ema_ok, ema_mult, ema_warn = self._cond4_daily_ema(ms)
        if ema_warn:
            warnings.append(ema_warn)
        multiplier *= ema_mult

        # ------ 조건 5: 멀티타임프레임 추세 합의 ------
        mtf_ok, mtf_warn = self._cond5_multi_timeframe(ms)
        if mtf_warn:
            warnings.append(mtf_warn)
        if not mtf_ok:
            return self._build_result(
                ms, coin, tradeable=False,
                multiplier=0.0, warnings=warnings,
                regime_strategy=regime,
            )

        # ------ 조건 6: 거래량 ------
        vol_ok, vol_warn = self._cond6_volume(ms)
        if vol_warn:
            warnings.append(vol_warn)
        if not vol_ok:
            return self._build_result(
                ms, coin, tradeable=False,
                multiplier=0.0, warnings=warnings,
                regime_strategy=regime,
            )

        # ------ 조건 7: 온체인 유입 ------
        oc_warn = self._cond7_onchain(ms, coin)
        if oc_warn:
            warnings.append(oc_warn)
            multiplier *= 0.8   # 강한 경고이지만 차단은 아님

        # ------ 조건 8: 감성 비토 ------
        sent_ok, sent_mult, sent_warn = self._cond8_sentiment(ms, regime)
        if sent_warn:
            warnings.append(sent_warn)
        multiplier *= sent_mult
        if not sent_ok:
            return self._build_result(
                ms, coin, tradeable=False,
                multiplier=0.0, warnings=warnings,
                regime_strategy=regime,
            )

        # ------ 조건 9: 코인 클러스터 (Phase C만) ------
        if self._phase_c:
            cluster_ok, cluster_warn = self._cond9_cluster(coin)
            if cluster_warn:
                warnings.append(cluster_warn)
            if not cluster_ok:
                return self._build_result(
                    ms, coin, tradeable=False,
                    multiplier=0.0, warnings=warnings,
                    regime_strategy=regime,
                )

        # ------ 조건 10: API 레이턴시 ------
        lat_ok, lat_warn = self._cond10_latency(ms)
        if lat_warn:
            warnings.append(lat_warn)
        if not lat_ok:
            return self._build_result(
                ms, coin, tradeable=False,
                multiplier=0.0, warnings=warnings,
                regime_strategy=regime,
                api_latency_ok=False,
            )

        # ------ 김치프리미엄 추가 필터 ------
        kimchi_mult, kimchi_warn = self._kimchi_premium_filter(ms)
        if kimchi_warn:
            warnings.append(kimchi_warn)
        if kimchi_mult == 0.0:
            return self._build_result(
                ms, coin, tradeable=False,
                multiplier=0.0, warnings=warnings,
                regime_strategy=regime,
            )
        multiplier *= kimchi_mult

        # ------ 조정 vs 반전 감지 ------
        pullback = self._detect_pullback(ms)
        reversal = self._detect_reversal(ms)

        result = self._build_result(
            ms, coin, tradeable=True,
            multiplier=multiplier,
            warnings=warnings,
            regime_strategy=regime,
            pullback=pullback,
            reversal=reversal,
        )

        await self._save_log(result)
        return result

    # ------------------------------------------------------------------
    # 조건 0: 서킷브레이커
    # ------------------------------------------------------------------

    def _cond0_circuit_breaker(self) -> tuple[bool, int, str]:
        """(ok, cb_level, warning_msg)."""
        level = self._cb.level
        if level == 0:
            return True, 0, ""
        if level >= 2:
            return False, level, f"서킷브레이커 Level {level} — 전 거래 중단"
        # Level 1: 매수만 차단
        return False, level, f"서킷브레이커 Level {level} — 매수 정지"

    # ------------------------------------------------------------------
    # 조건 1: ADX 신뢰도 → 레짐 결정
    # ------------------------------------------------------------------

    def _cond1_adx_regime(
        self, ms: MarketState
    ) -> tuple[bool, str, str]:
        """(ok, warning_msg, regime_strategy).

        Phase A/B: ADX 단독.
        Phase C: HMM 신뢰도 × ADX 복합 (phase_c_enabled 플래그).
        """
        adx = ms.adx_5m

        # Phase C: HMM 레짐 활용
        if self._phase_c and ms.hmm_regime >= 0:
            regime = self._hmm_to_regime(ms.hmm_regime, adx, ms.fear_greed)
            if adx < ADX_MIN_THRESHOLD:
                return False, f"ADX {adx:.1f} < {ADX_MIN_THRESHOLD} — 추세 약함", regime
            return True, "", regime

        # Phase A/B: ADX 단독
        if adx < ADX_MIN_THRESHOLD:
            return False, f"ADX {adx:.1f} < {ADX_MIN_THRESHOLD} — 매수 보류", "HOLD"

        regime = self._adx_to_regime(adx, ms.fear_greed)
        return True, "", regime

    @staticmethod
    def _adx_to_regime(adx: float, fear_greed: float) -> str:
        if adx >= ADX_STRONG_THRESHOLD:
            return "TREND_STRONG"
        if adx >= ADX_NORMAL_THRESHOLD:
            return "TREND_NORMAL"
        if fear_greed < 30:
            return "DCA"
        return "GRID"

    @staticmethod
    def _hmm_to_regime(hmm_regime: int, adx: float, fear_greed: float) -> str:
        if hmm_regime == 0:
            return "TREND_STRONG" if adx > ADX_STRONG_THRESHOLD else "TREND_NORMAL"
        if hmm_regime == 1:
            return "TREND_NORMAL"
        if hmm_regime == 2:
            return "GRID"
        # hmm_regime == 3
        if fear_greed < 30:
            return "DCA"
        return "HOLD"

    # ------------------------------------------------------------------
    # 조건 2: Fear&Greed
    # ------------------------------------------------------------------

    def _cond2_fear_greed(self, ms: MarketState) -> tuple[bool, str]:
        fg = ms.fear_greed
        if fg < FEAR_GREED_MIN:
            return False, f"Fear&Greed {fg:.0f} < {FEAR_GREED_MIN} — 극단적 공포"
        if fg > FEAR_GREED_MAX:
            return False, f"Fear&Greed {fg:.0f} > {FEAR_GREED_MAX} — 극단적 탐욕"
        return True, ""

    # ------------------------------------------------------------------
    # 조건 3: BTC 도미넌스
    # ------------------------------------------------------------------

    def _cond3_btc_dominance(
        self, ms: MarketState
    ) -> tuple[bool, float, str]:
        """(ok=항상True, multiplier, warning)."""
        dom = ms.btc_dominance
        if dom > BTC_DOM_THRESHOLD:
            return True, 0.7, f"BTC 도미넌스 {dom:.1f}% > {BTC_DOM_THRESHOLD}% → 신호 ×0.7"
        return True, 1.0, ""

    # ------------------------------------------------------------------
    # 조건 4: 일봉 EMA50 vs EMA200
    # ------------------------------------------------------------------

    def _cond4_daily_ema(
        self, ms: MarketState
    ) -> tuple[bool, float, str]:
        """데드크로스(EMA50<EMA200) → ×0.8, 골든크로스 → ×1.0."""
        ema50 = ms.ema50_1d
        ema200 = ms.ema200_1d
        if ema50 <= 0 or ema200 <= 0:
            return True, 1.0, ""  # 데이터 없으면 패스
        if ema50 < ema200:
            return True, 0.8, f"일봉 데드크로스 (EMA50={ema50:.0f} < EMA200={ema200:.0f}) → ×0.8"
        return True, 1.0, ""

    # ------------------------------------------------------------------
    # 조건 5: 멀티타임프레임 추세 합의
    # ------------------------------------------------------------------

    def _cond5_multi_timeframe(self, ms: MarketState) -> tuple[bool, str]:
        """5분봉/1시간봉/일봉 추세 합의.

        3개 중 2개 이상 불일치 → 매수 보류.
        3개 모두 불일치 → 즉시 차단.
        """
        results: list[bool] = []
        details: list[str] = []

        # 5a: 5분봉 EMA7 > EMA25
        r5a = ms.ema7_5m > ms.ema25_5m
        results.append(r5a)
        details.append(f"5m EMA7{'>'if r5a else '<'}EMA25")

        # 5b: 1시간봉 EMA20 > EMA50
        r5b = ms.ema20_1h > ms.ema50_1h
        results.append(r5b)
        details.append(f"1h EMA20{'>'if r5b else '<'}EMA50")

        # 5c: 일봉 EMA50 > EMA200 (shift(1) 적용값)
        r5c = ms.ema50_1d > ms.ema200_1d if (ms.ema50_1d > 0 and ms.ema200_1d > 0) else True
        results.append(r5c)
        details.append(f"1d EMA50{'>'if r5c else '<'}EMA200")

        true_count = sum(results)
        false_count = 3 - true_count

        if false_count == 3:
            return False, f"MTF 3개 모두 하락 추세 — 즉시 차단 [{', '.join(details)}]"
        if false_count >= 2:
            return False, f"MTF {false_count}개 불일치 — 매수 보류 [{', '.join(details)}]"
        return True, ""

    # ------------------------------------------------------------------
    # 조건 6: 거래량
    # ------------------------------------------------------------------

    def _cond6_volume(self, ms: MarketState) -> tuple[bool, str]:
        """5분봉 거래량 ≥ 20기간 평균 × 50%."""
        ratio = ms.volume_ratio_5m
        if ratio < VOLUME_RATIO_MIN:
            return False, f"거래량 부족 (ratio={ratio:.2f} < {VOLUME_RATIO_MIN})"
        return True, ""

    # ------------------------------------------------------------------
    # 조건 7: 온체인 거래소 유입
    # ------------------------------------------------------------------

    def _cond7_onchain(self, ms: MarketState, coin: str) -> str:
        """전일 대비 +50% 초과 시 경고 (차단 아님)."""
        inflow = ms.exchange_inflow
        prev = self._prev_onchain.get(coin, 0.0)
        if prev > 0 and inflow > 0:
            ratio = inflow / prev
            if ratio >= ONCHAIN_INFLOW_SURGE:
                return f"온체인 유입 전일比 +{(ratio-1)*100:.0f}% 급증 — 매도 압력 경고"
        return ""

    def update_prev_onchain(self, coin: str, inflow: float) -> None:
        """전일 온체인 유입 업데이트 (매일 호출)."""
        self._prev_onchain[coin] = inflow

    # ------------------------------------------------------------------
    # 조건 8: 감성 비토
    # ------------------------------------------------------------------

    def _cond8_sentiment(
        self, ms: MarketState, regime: str
    ) -> tuple[bool, float, str]:
        """VADER+FinBERT+시장지수 3방향 일치 시만 비토.

        레짐 연동:
          레짐0,1 + 부정 → ×0.5 (차단 아님)
          레짐2,3 + 부정 → USDT 전환 (차단)
          레짐0,1 + 긍정 → 풀포지션 ×1.0
          레짐2,3 + 긍정 → 반등 대기 ×0.5
        """
        score = ms.sentiment_score
        conf = ms.sentiment_confidence
        fg = ms.fear_greed

        # 시장지수 방향
        market_negative = fg < 30
        market_positive = fg > 60

        # 부정 3방향 합의 (score<−0.3, conf>0.5, market_negative)
        is_negative_consensus = (
            score < SENTIMENT_VETO_THRESHOLD
            and conf > 0.5
            and market_negative
        )

        is_uptrend_regime = regime in ("TREND_STRONG", "TREND_NORMAL")

        if is_negative_consensus:
            if is_uptrend_regime:
                return True, 0.5, f"부정 감성 비토 (레짐 상승) → ×0.5 (score={score:.2f})"
            else:
                return False, 0.0, f"부정 감성 비토 (레짐 하락) → USDT 전환 (score={score:.2f})"

        # 하락 레짐 + 긍정 감성 → 반등 대기
        if not is_uptrend_regime and score > 0.3 and market_positive:
            return True, 0.5, f"하락 레짐 + 긍정 감성 → 반등 대기 ×0.5"

        return True, 1.0, ""

    # ------------------------------------------------------------------
    # 조건 9: 코인 클러스터 (Phase C)
    # ------------------------------------------------------------------

    def _cond9_cluster(self, coin: str) -> tuple[bool, str]:
        """상관 >0.8 동일 클러스터 코인 중복 보유 방지 (Phase C 구현 예정)."""
        # Phase C에서 layer0_5_cluster.py 연동 예정
        return True, ""

    # ------------------------------------------------------------------
    # 조건 10: API 레이턴시
    # ------------------------------------------------------------------

    def _cond10_latency(self, ms: MarketState) -> tuple[bool, str]:
        lat = ms.api_latency_ms
        if lat > LATENCY_MAX_MS:
            return False, f"API 레이턴시 {lat:.0f}ms > {LATENCY_MAX_MS:.0f}ms — 해당 코인 스킵"
        return True, ""

    # ------------------------------------------------------------------
    # 김치프리미엄 추가 필터
    # ------------------------------------------------------------------

    def _kimchi_premium_filter(self, ms: MarketState) -> tuple[float, str]:
        """(multiplier, warning).

        >5%  → 과열 경고 ×0.8
        <-2% → 역프리미엄 보류 (차단)
        """
        kp = ms.kimchi_premium
        if kp > KIMCHI_PREMIUM_HIGH:
            return 0.8, f"김치프리미엄 {kp:.2f}% > {KIMCHI_PREMIUM_HIGH}% — 과열 ×0.8"
        if kp < KIMCHI_PREMIUM_LOW:
            return 0.0, f"역프리미엄 {kp:.2f}% < {KIMCHI_PREMIUM_LOW}% — 매수 보류"
        return 1.0, ""

    # ------------------------------------------------------------------
    # 조정 / 반전 감지 (추세 추종 전략 연동)
    # ------------------------------------------------------------------

    def _detect_pullback(self, ms: MarketState) -> bool:
        """조정 감지 (홀딩 신호).

        조건: ADX>25 AND SuperTrend미반전 AND EMA99위 AND RSI≥40
        """
        return (
            ms.adx_5m > 25
            and ms.supertrend_signal == 1         # 상승 방향 유지
            and ms.close_5m > ms.ema99_5m         # EMA99 위
            and ms.rsi_5m >= 40
        )

    def _detect_reversal(self, ms: MarketState) -> bool:
        """반전 감지 (청산 신호).

        조건: ADX<20 OR SuperTrend반전 OR EMA99이탈 OR RSI<40
        """
        return (
            ms.adx_5m < 20
            or ms.supertrend_signal == -1
            or ms.close_5m < ms.ema99_5m
            or ms.rsi_5m < 40
        )

    # ------------------------------------------------------------------
    # FilterResult 생성
    # ------------------------------------------------------------------

    def _build_result(
        self,
        ms: MarketState,
        coin: str,
        tradeable: bool,
        multiplier: float,
        warnings: list[str],
        regime_strategy: str = "HOLD",
        cb_level: int = 0,
        api_latency_ok: bool = True,
        pullback: bool = False,
        reversal: bool = False,
    ) -> FilterResult:
        return FilterResult(
            coin=coin,
            timestamp=datetime.now(timezone.utc),
            tradeable=tradeable,
            regime_strategy=regime_strategy,
            signal_multiplier=round(multiplier, 4),
            adx_value=ms.adx_5m,
            supertrend_direction=ms.supertrend_signal,
            atr_value=ms.atr_5m,
            active_warnings=warnings,
            pullback_detected=pullback,
            reversal_detected=reversal,
            api_latency_ok=api_latency_ok,
            daily_loss_pct=0.0,
            circuit_breaker_level=cb_level,
        )

    # ------------------------------------------------------------------
    # SQLite 저장
    # ------------------------------------------------------------------

    async def _save_log(self, result: FilterResult) -> None:
        if not self._cache:
            return
        try:
            row = {
                "coin": result.coin,
                "timestamp": result.timestamp.isoformat(),
                "tradeable": int(result.tradeable),
                "regime_strategy": result.regime_strategy,
                "signal_multiplier": result.signal_multiplier,
                "adx_value": result.adx_value,
                "supertrend_direction": result.supertrend_direction,
                "atr_value": result.atr_value,
                "active_warnings": json.dumps(result.active_warnings, ensure_ascii=False),
                "pullback_detected": int(result.pullback_detected),
                "reversal_detected": int(result.reversal_detected),
                "api_latency_ms": result.api_latency_ok * 0.0,  # 실제 latency는 ms에서 가져옴
                "daily_loss_pct": result.daily_loss_pct,
                "circuit_breaker_level": result.circuit_breaker_level,
            }
            self._cache.upsert_candle("layer1_log", result.coin, row)
        except Exception as exc:
            logger.error("layer1_log 저장 실패: %s", exc)

    # ------------------------------------------------------------------
    # 저유동성 사전 제외
    # ------------------------------------------------------------------

    @staticmethod
    def is_low_liquidity(volume_24h_krw: float) -> bool:
        """24h 거래량 < 1억원 → True (PairlistManager에서도 제외하지만 2중 체크)."""
        return volume_24h_krw < 100_000_000


# ------------------------------------------------------------------
# 배치 체크 헬퍼 (여러 코인 asyncio.gather)
# ------------------------------------------------------------------

async def check_all(
    filt: Layer1MarketFilter,
    market_states: dict[str, MarketState],
) -> dict[str, FilterResult]:
    """여러 코인 FilterResult 병렬 생성.

    사용법:
        results = await check_all(filt, {coin: ms for coin, ms in states.items()})
    """
    import asyncio

    async def _check_one(coin: str, ms: MarketState):
        return coin, await filt.check(ms, coin)

    tasks = [_check_one(c, ms) for c, ms in market_states.items()]
    pairs = await asyncio.gather(*tasks, return_exceptions=False)
    return dict(pairs)
