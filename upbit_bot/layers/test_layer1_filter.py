"""
test_layer1_filter.py — Layer1MarketFilter + CircuitBreaker 단위 테스트
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from layers.layer1_filter import (
    ADX_MIN_THRESHOLD,
    BTC_DOM_THRESHOLD,
    FEAR_GREED_MAX,
    FEAR_GREED_MIN,
    KIMCHI_PREMIUM_HIGH,
    KIMCHI_PREMIUM_LOW,
    LATENCY_MAX_MS,
    Layer1MarketFilter,
    check_all,
)
from risk.circuit_breaker import CircuitBreaker
from schema import MarketState


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _ms(**kwargs) -> MarketState:
    """기본값이 '모든 조건 통과'인 MarketState 생성."""
    defaults = dict(
        coin="KRW-BTC",
        timestamp=datetime.now(timezone.utc),
        # 5분봉
        close_5m=90_000_000.0,
        open_5m=89_000_000.0,
        high_5m=91_000_000.0,
        low_5m=88_000_000.0,
        volume_5m=100.0,
        rsi_5m=55.0,
        ema7_5m=90_500_000.0,   # EMA7 > EMA25 (상승)
        ema25_5m=89_000_000.0,
        ema99_5m=85_000_000.0,
        volume_ratio_5m=1.2,    # 20기간 평균 120%
        adx_5m=25.0,            # 정상 ADX
        supertrend_signal=1,
        atr_5m=500_000.0,
        # 1시간봉
        ema20_1h=90_000_000.0,
        ema50_1h=88_000_000.0,  # EMA20 > EMA50 (상승)
        trend_dir_1h=1,
        # 일봉 (shift(1) 적용값)
        ema50_1d=80_000_000.0,
        ema200_1d=75_000_000.0, # EMA50 > EMA200 (골든크로스)
        rsi_1d=55.0,
        trend_encoding_1d=1,
        # 시장지수
        fear_greed=50.0,        # 정상 범위
        btc_dominance=45.0,     # 60% 미만
        altcoin_season=60.0,
        # 온체인
        exchange_inflow=100.0,
        exchange_outflow=80.0,
        # 감성
        sentiment_score=0.1,    # 중립~약한 긍정
        sentiment_confidence=0.3,
        # 추세/레짐
        adx_1h=22.0,
        hmm_regime=-1,          # Phase A (HMM 미적용)
        # 업비트 특화
        kimchi_premium=1.0,     # 정상 범위
        obi=0.1,
        top5_concentration=0.4,
        orderbook_wall_ratio=2.0,
        api_latency_ms=100.0,   # 500ms 미만
        # 마이크로스트럭처
        tick_imbalance=0.1,
        trade_velocity=1.0,
        macd_5m=0.0,
        macd_signal_5m=0.0,
        bb_upper_5m=0.0,
        bb_lower_5m=0.0,
    )
    defaults.update(kwargs)
    return MarketState(**defaults)


def _cb(level: int = 0) -> CircuitBreaker:
    cb = CircuitBreaker()
    if level > 0:
        cb.trigger(level, "테스트용")
    return cb


def _filt(cb: CircuitBreaker | None = None, **kwargs) -> Layer1MarketFilter:
    return Layer1MarketFilter(circuit_breaker=cb or _cb(), **kwargs)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_initial_level_zero(self):
        cb = CircuitBreaker()
        assert cb.level == 0
        assert not cb.is_buy_blocked()
        assert not cb.is_all_blocked()

    def test_trigger_level1_blocks_buy(self):
        cb = CircuitBreaker()
        cb.trigger(1, "테스트")
        assert cb.level == 1
        assert cb.is_buy_blocked()
        assert not cb.is_all_blocked()

    def test_trigger_level2_blocks_all(self):
        cb = CircuitBreaker()
        cb.trigger(2, "테스트")
        assert cb.is_all_blocked()

    def test_higher_level_overwrites(self):
        cb = CircuitBreaker()
        cb.trigger(1, "낮음")
        cb.trigger(3, "높음")
        assert cb.level == 3

    def test_lower_level_does_not_overwrite(self):
        cb = CircuitBreaker()
        cb.trigger(3, "높음")
        cb.trigger(1, "낮음")
        assert cb.level == 3

    def test_api_error_trigger_at_3(self):
        cb = CircuitBreaker()
        cb.record_api_error()
        cb.record_api_error()
        assert cb.level == 0
        cb.record_api_error()
        assert cb.level == 4

    def test_reset(self):
        cb = CircuitBreaker()
        cb.trigger(3, "테스트")
        cb.reset()
        assert cb.level == 0

    def test_manual_required_level5(self):
        cb = CircuitBreaker()
        cb.trigger(5, "24h -15%")
        assert cb.is_manual_required()

    def test_price_drop_1m_triggers_level1(self):
        cb = CircuitBreaker()
        cb.check_price_drop(
            coin="KRW-BTC",
            price_1m_ago=100.0,
            price_10m_ago=100.0,
            current_price=96.0,    # -4%
            daily_loss_pct=-0.02,
            cumulative_loss_24h=-0.02,
        )
        assert cb.level >= 1

    def test_price_drop_10m_triggers_level2(self):
        cb = CircuitBreaker()
        cb.check_price_drop(
            coin="KRW-BTC",
            price_1m_ago=100.0,
            price_10m_ago=100.0,
            current_price=91.0,   # -9%
            daily_loss_pct=-0.02,
            cumulative_loss_24h=-0.02,
        )
        assert cb.level >= 2

    def test_get_history(self):
        cb = CircuitBreaker()
        cb.trigger(1, "이벤트1")
        cb.trigger(3, "이벤트2")
        hist = cb.get_history()
        assert len(hist) == 2


# ---------------------------------------------------------------------------
# Layer1: 조건 0 — 서킷브레이커
# ---------------------------------------------------------------------------

class TestCond0CircuitBreaker:
    def test_cb_level0_passes(self):
        filt = _filt(_cb(0))
        result = run(filt.check(_ms(), "KRW-BTC"))
        assert result.circuit_breaker_level == 0

    def test_cb_level1_blocks(self):
        filt = _filt(_cb(1))
        result = run(filt.check(_ms(), "KRW-BTC"))
        assert result.tradeable is False
        assert result.circuit_breaker_level == 1

    def test_cb_level3_blocks(self):
        filt = _filt(_cb(3))
        result = run(filt.check(_ms(), "KRW-BTC"))
        assert result.tradeable is False
        assert any("서킷브레이커" in w for w in result.active_warnings)


# ---------------------------------------------------------------------------
# Layer1: 조건 1 — ADX
# ---------------------------------------------------------------------------

class TestCond1ADX:
    def test_adx_below_min_blocks(self):
        filt = _filt()
        ms = _ms(adx_5m=ADX_MIN_THRESHOLD - 1)
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.tradeable is False
        assert any("ADX" in w for w in result.active_warnings)

    def test_adx_above_strong_gives_trend_strong(self):
        filt = _filt()
        ms = _ms(adx_5m=35.0, fear_greed=50.0)
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.tradeable is True
        assert result.regime_strategy == "TREND_STRONG"

    def test_adx_normal_gives_trend_normal(self):
        filt = _filt()
        ms = _ms(adx_5m=25.0, fear_greed=50.0)
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.regime_strategy == "TREND_NORMAL"

    def test_adx_low_fg_low_gives_dca(self):
        filt = _filt()
        ms = _ms(adx_5m=18.0, fear_greed=20.0)
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.regime_strategy == "DCA"


# ---------------------------------------------------------------------------
# Layer1: 조건 2 — Fear&Greed
# ---------------------------------------------------------------------------

class TestCond2FearGreed:
    def test_fg_below_min_blocks(self):
        filt = _filt()
        ms = _ms(fear_greed=FEAR_GREED_MIN - 1)
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.tradeable is False
        assert any("Fear&Greed" in w for w in result.active_warnings)

    def test_fg_above_max_blocks(self):
        filt = _filt()
        ms = _ms(fear_greed=FEAR_GREED_MAX + 1)
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.tradeable is False

    def test_fg_normal_passes(self):
        filt = _filt()
        ms = _ms(fear_greed=50.0)
        result = run(filt.check(ms, "KRW-BTC"))
        # Fear&Greed는 통과하되 다른 조건에 의해 결과 달라질 수 있음
        assert not any("Fear&Greed" in w for w in result.active_warnings)


# ---------------------------------------------------------------------------
# Layer1: 조건 3 — BTC 도미넌스
# ---------------------------------------------------------------------------

class TestCond3BtcDominance:
    def test_high_dominance_reduces_multiplier(self):
        filt = _filt()
        ms = _ms(btc_dominance=BTC_DOM_THRESHOLD + 5)
        result = run(filt.check(ms, "KRW-BTC"))
        # tradeable일 수 있지만 multiplier 감소
        if result.tradeable:
            assert result.signal_multiplier < 1.0
        assert any("도미넌스" in w for w in result.active_warnings)

    def test_normal_dominance_no_penalty(self):
        filt = _filt()
        ms = _ms(btc_dominance=45.0)
        result = run(filt.check(ms, "KRW-BTC"))
        assert not any("도미넌스" in w for w in result.active_warnings)


# ---------------------------------------------------------------------------
# Layer1: 조건 4 — 일봉 EMA 크로스
# ---------------------------------------------------------------------------

class TestCond4DailyEma:
    def test_dead_cross_reduces_multiplier(self):
        filt = _filt()
        ms = _ms(ema50_1d=70_000_000.0, ema200_1d=80_000_000.0)  # 데드크로스
        result = run(filt.check(ms, "KRW-BTC"))
        if result.tradeable:
            assert result.signal_multiplier < 1.0
        assert any("데드크로스" in w for w in result.active_warnings)

    def test_golden_cross_no_penalty(self):
        filt = _filt()
        ms = _ms(ema50_1d=90_000_000.0, ema200_1d=70_000_000.0)  # 골든크로스
        result = run(filt.check(ms, "KRW-BTC"))
        assert not any("데드크로스" in w for w in result.active_warnings)


# ---------------------------------------------------------------------------
# Layer1: 조건 5 — 멀티타임프레임 추세 합의
# ---------------------------------------------------------------------------

class TestCond5MultiTimeframe:
    def test_all_bearish_blocks(self):
        """3개 모두 하락 → 즉시 차단."""
        filt = _filt()
        ms = _ms(
            ema7_5m=80_000_000.0, ema25_5m=90_000_000.0,  # 5m 하락
            ema20_1h=80_000_000.0, ema50_1h=90_000_000.0,  # 1h 하락
            ema50_1d=60_000_000.0, ema200_1d=80_000_000.0, # 1d 하락
        )
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.tradeable is False
        assert any("MTF" in w for w in result.active_warnings)

    def test_two_bearish_blocks(self):
        """2개 하락 → 매수 보류."""
        filt = _filt()
        ms = _ms(
            ema7_5m=80_000_000.0, ema25_5m=90_000_000.0,  # 5m 하락
            ema20_1h=80_000_000.0, ema50_1h=90_000_000.0,  # 1h 하락
            ema50_1d=90_000_000.0, ema200_1d=70_000_000.0, # 1d 상승
        )
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.tradeable is False

    def test_one_bearish_passes(self):
        """1개 하락 → 통과 (나머지 조건 충족 시)."""
        filt = _filt()
        ms = _ms(
            ema7_5m=80_000_000.0, ema25_5m=90_000_000.0,  # 5m 하락만
            ema20_1h=90_000_000.0, ema50_1h=80_000_000.0,  # 1h 상승
            ema50_1d=90_000_000.0, ema200_1d=70_000_000.0, # 1d 상승
        )
        result = run(filt.check(ms, "KRW-BTC"))
        # MTF 조건 자체는 통과 (1개 불일치)
        assert not any("MTF" in w for w in result.active_warnings)

    def test_all_bullish_passes(self):
        """3개 모두 상승 → MTF 조건 통과."""
        filt = _filt()
        ms = _ms(
            ema7_5m=91_000_000.0, ema25_5m=89_000_000.0,
            ema20_1h=91_000_000.0, ema50_1h=89_000_000.0,
            ema50_1d=90_000_000.0, ema200_1d=70_000_000.0,
        )
        result = run(filt.check(ms, "KRW-BTC"))
        assert not any("MTF" in w for w in result.active_warnings)


# ---------------------------------------------------------------------------
# Layer1: 조건 6 — 거래량
# ---------------------------------------------------------------------------

class TestCond6Volume:
    def test_low_volume_blocks(self):
        filt = _filt()
        ms = _ms(volume_ratio_5m=0.3)  # 30% → 50% 미만
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.tradeable is False
        assert any("거래량" in w for w in result.active_warnings)

    def test_sufficient_volume_passes(self):
        filt = _filt()
        ms = _ms(volume_ratio_5m=1.5)
        result = run(filt.check(ms, "KRW-BTC"))
        assert not any("거래량" in w for w in result.active_warnings)


# ---------------------------------------------------------------------------
# Layer1: 조건 7 — 온체인 유입
# ---------------------------------------------------------------------------

class TestCond7Onchain:
    def test_inflow_surge_warning(self):
        filt = _filt(prev_onchain={"KRW-BTC": 100.0})
        ms = _ms(exchange_inflow=200.0)  # 전일比 +100%
        result = run(filt.check(ms, "KRW-BTC"))
        # 차단은 아니지만 경고 존재
        assert any("온체인" in w for w in result.active_warnings)

    def test_normal_inflow_no_warning(self):
        filt = _filt(prev_onchain={"KRW-BTC": 100.0})
        ms = _ms(exchange_inflow=120.0)  # +20%
        result = run(filt.check(ms, "KRW-BTC"))
        assert not any("온체인" in w for w in result.active_warnings)


# ---------------------------------------------------------------------------
# Layer1: 조건 8 — 감성 비토
# ---------------------------------------------------------------------------

class TestCond8Sentiment:
    def test_negative_consensus_uptrend_reduces_multiplier(self):
        """부정 3방향 합의 + 상승 레짐 → ×0.5 (차단 아님)."""
        filt = _filt()
        ms = _ms(
            adx_5m=35.0,            # TREND_STRONG 레짐
            sentiment_score=-0.5,
            sentiment_confidence=0.8,
            fear_greed=20.0,        # 공포 → 시장지수 부정
        )
        result = run(filt.check(ms, "KRW-BTC"))
        if result.tradeable:
            assert result.signal_multiplier < 1.0
            assert any("감성" in w for w in result.active_warnings)

    def test_negative_consensus_downtrend_blocks(self):
        """부정 3방향 합의 + 하락 레짐 → USDT 전환 (차단)."""
        filt = _filt()
        ms = _ms(
            adx_5m=18.0,            # GRID 레짐
            fear_greed=20.0,        # DCA 레짐
            sentiment_score=-0.6,
            sentiment_confidence=0.9,
        )
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.tradeable is False
        assert any("감성" in w for w in result.active_warnings)

    def test_neutral_sentiment_passes(self):
        filt = _filt()
        ms = _ms(sentiment_score=0.1, sentiment_confidence=0.3)
        result = run(filt.check(ms, "KRW-BTC"))
        assert not any("감성" in w for w in result.active_warnings)


# ---------------------------------------------------------------------------
# Layer1: 조건 10 — API 레이턴시
# ---------------------------------------------------------------------------

class TestCond10Latency:
    def test_high_latency_blocks(self):
        filt = _filt()
        ms = _ms(api_latency_ms=LATENCY_MAX_MS + 100)
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.tradeable is False
        assert result.api_latency_ok is False
        assert any("레이턴시" in w for w in result.active_warnings)

    def test_low_latency_passes(self):
        filt = _filt()
        ms = _ms(api_latency_ms=100.0)
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.api_latency_ok is True


# ---------------------------------------------------------------------------
# 김치프리미엄 필터
# ---------------------------------------------------------------------------

class TestKimchiPremiumFilter:
    def test_high_premium_reduces_multiplier(self):
        filt = _filt()
        ms = _ms(kimchi_premium=KIMCHI_PREMIUM_HIGH + 1)
        result = run(filt.check(ms, "KRW-BTC"))
        if result.tradeable:
            assert result.signal_multiplier < 1.0
        assert any("김치프리미엄" in w for w in result.active_warnings)

    def test_reverse_premium_blocks(self):
        filt = _filt()
        ms = _ms(kimchi_premium=KIMCHI_PREMIUM_LOW - 1)
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.tradeable is False
        assert any("역프리미엄" in w for w in result.active_warnings)

    def test_normal_premium_passes(self):
        filt = _filt()
        ms = _ms(kimchi_premium=1.5)
        result = run(filt.check(ms, "KRW-BTC"))
        assert not any("프리미엄" in w for w in result.active_warnings)


# ---------------------------------------------------------------------------
# 조정 / 반전 감지
# ---------------------------------------------------------------------------

class TestPullbackReversal:
    def test_pullback_detected(self):
        filt = _filt()
        ms = _ms(
            adx_5m=28.0,
            supertrend_signal=1,
            close_5m=90_000_000.0,
            ema99_5m=85_000_000.0,
            rsi_5m=45.0,
        )
        assert filt._detect_pullback(ms) is True

    def test_reversal_detected_low_adx(self):
        filt = _filt()
        ms = _ms(adx_5m=15.0)
        assert filt._detect_reversal(ms) is True

    def test_reversal_detected_supertrend_flip(self):
        filt = _filt()
        ms = _ms(supertrend_signal=-1)
        assert filt._detect_reversal(ms) is True

    def test_no_reversal_when_all_ok(self):
        filt = _filt()
        ms = _ms(
            adx_5m=28.0,
            supertrend_signal=1,
            close_5m=90_000_000.0,
            ema99_5m=85_000_000.0,
            rsi_5m=55.0,
        )
        assert filt._detect_reversal(ms) is False


# ---------------------------------------------------------------------------
# 전체 통과 (모든 조건 충족)
# ---------------------------------------------------------------------------

class TestFullPass:
    def test_all_conditions_pass(self):
        filt = _filt()
        ms = _ms()
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.tradeable is True
        assert result.signal_multiplier > 0
        assert result.circuit_breaker_level == 0

    def test_multiplier_stacks(self):
        """도미넌스 초과 + 데드크로스 → multiplier 누적 감소."""
        filt = _filt()
        ms = _ms(
            btc_dominance=65.0,              # ×0.7
            ema50_1d=60_000_000.0,           # 데드크로스 ×0.8
            ema200_1d=80_000_000.0,
        )
        result = run(filt.check(ms, "KRW-BTC"))
        if result.tradeable:
            assert result.signal_multiplier == pytest.approx(0.7 * 0.8, rel=1e-3)


# ---------------------------------------------------------------------------
# 배치 체크
# ---------------------------------------------------------------------------

class TestCheckAll:
    def test_check_all_returns_all_coins(self):
        filt = _filt()
        states = {
            "KRW-BTC": _ms(coin="KRW-BTC"),
            "KRW-ETH": _ms(coin="KRW-ETH"),
        }
        results = run(check_all(filt, states))
        assert set(results.keys()) == {"KRW-BTC", "KRW-ETH"}

    def test_check_all_independent_results(self):
        """CB 없이 KRW-BTC 통과, KRW-ETH ADX 낮아 차단."""
        filt = _filt()
        states = {
            "KRW-BTC": _ms(coin="KRW-BTC", adx_5m=25.0),
            "KRW-ETH": _ms(coin="KRW-ETH", adx_5m=ADX_MIN_THRESHOLD - 1),
        }
        results = run(check_all(filt, states))
        assert results["KRW-BTC"].tradeable is True
        assert results["KRW-ETH"].tradeable is False


# ---------------------------------------------------------------------------
# FilterResult 구조
# ---------------------------------------------------------------------------

class TestFilterResultStructure:
    def test_result_has_all_fields(self):
        filt = _filt()
        result = run(filt.check(_ms(), "KRW-BTC"))
        assert hasattr(result, "tradeable")
        assert hasattr(result, "regime_strategy")
        assert hasattr(result, "signal_multiplier")
        assert hasattr(result, "adx_value")
        assert hasattr(result, "supertrend_direction")
        assert hasattr(result, "atr_value")
        assert hasattr(result, "active_warnings")
        assert hasattr(result, "pullback_detected")
        assert hasattr(result, "reversal_detected")
        assert hasattr(result, "api_latency_ok")
        assert hasattr(result, "circuit_breaker_level")

    def test_regime_strategy_valid_values(self):
        valid = {"TREND_STRONG", "TREND_NORMAL", "GRID", "DCA", "HOLD"}
        filt = _filt()
        result = run(filt.check(_ms(), "KRW-BTC"))
        assert result.regime_strategy in valid

    def test_multiplier_in_range(self):
        filt = _filt()
        result = run(filt.check(_ms(), "KRW-BTC"))
        assert 0.0 <= result.signal_multiplier <= 1.5


# ---------------------------------------------------------------------------
# Phase C HMM 연동 (스킵 플래그 검증)
# ---------------------------------------------------------------------------

class TestPhaseC:
    def test_phase_a_skips_hmm(self):
        """Phase A: hmm_regime=-1이어도 ADX로 정상 처리."""
        filt = Layer1MarketFilter(circuit_breaker=_cb(), phase_c_enabled=False)
        ms = _ms(hmm_regime=-1, adx_5m=25.0)
        result = run(filt.check(ms, "KRW-BTC"))
        # HMM 없이 ADX로 레짐 결정
        assert result.regime_strategy in {"TREND_NORMAL", "TREND_STRONG", "GRID", "DCA"}

    def test_phase_c_uses_hmm(self):
        """Phase C: hmm_regime 값으로 레짐 결정."""
        filt = Layer1MarketFilter(circuit_breaker=_cb(), phase_c_enabled=True)
        ms = _ms(hmm_regime=0, adx_5m=35.0)   # 레짐0 + ADX 강함
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.regime_strategy == "TREND_STRONG"

    def test_phase_c_regime3_fg_low_gives_dca(self):
        filt = Layer1MarketFilter(circuit_breaker=_cb(), phase_c_enabled=True)
        ms = _ms(hmm_regime=3, adx_5m=20.0, fear_greed=20.0)
        result = run(filt.check(ms, "KRW-BTC"))
        assert result.regime_strategy == "DCA"
