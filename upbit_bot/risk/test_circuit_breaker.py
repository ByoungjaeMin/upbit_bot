"""risk/test_circuit_breaker.py — CircuitBreaker 단위 테스트."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from risk.circuit_breaker import CircuitBreaker, TriggerEvent


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _cb() -> CircuitBreaker:
    return CircuitBreaker()


# ---------------------------------------------------------------------------
# 초기 상태
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_level_zero(self):
        assert _cb().level == 0

    def test_buy_not_blocked(self):
        assert _cb().is_buy_blocked() is False

    def test_all_not_blocked(self):
        assert _cb().is_all_blocked() is False

    def test_manual_not_required(self):
        assert _cb().is_manual_required() is False

    def test_get_status_level_zero(self):
        status = _cb().get_status()
        assert status["level"] == 0
        assert status["triggered_at"] is None
        assert status["api_error_count"] == 0


# ---------------------------------------------------------------------------
# trigger()
# ---------------------------------------------------------------------------

class TestTrigger:
    def test_level1_sets_level(self):
        cb = _cb()
        cb.trigger(1, "테스트")
        assert cb.level == 1

    def test_level1_blocks_buy(self):
        cb = _cb()
        cb.trigger(1, "테스트")
        assert cb.is_buy_blocked() is True

    def test_level1_does_not_block_all(self):
        cb = _cb()
        cb.trigger(1, "테스트")
        assert cb.is_all_blocked() is False

    def test_level2_blocks_all(self):
        cb = _cb()
        cb.trigger(2, "테스트")
        assert cb.is_all_blocked() is True

    def test_level5_manual_required(self):
        cb = _cb()
        cb.trigger(5, "24h -15%")
        assert cb.is_manual_required() is True

    def test_higher_level_overwrites(self):
        cb = _cb()
        cb.trigger(1, "낮음")
        cb.trigger(3, "높음")
        assert cb.level == 3

    def test_lower_level_does_not_overwrite(self):
        cb = _cb()
        cb.trigger(3, "높음")
        cb.trigger(1, "낮음")
        assert cb.level == 3

    def test_invalid_level_raises(self):
        cb = _cb()
        with pytest.raises(ValueError):
            cb.trigger(0, "잘못된 레벨")
        with pytest.raises(ValueError):
            cb.trigger(6, "잘못된 레벨")

    def test_trigger_records_history(self):
        cb = _cb()
        cb.trigger(1, "이벤트A")
        cb.trigger(2, "이벤트B")
        history = cb.get_history()
        assert len(history) == 2
        assert isinstance(history[0], TriggerEvent)
        assert history[0].level == 1
        assert history[1].level == 2

    def test_get_history_last_n(self):
        cb = _cb()
        for i in range(1, 6):
            cb.trigger(1, f"이벤트{i}")
            # 같은 레벨은 덮어쓰지 않으므로 reset 필요
            cb.reset()
        history = cb.get_history(last_n=3)
        assert len(history) == 3


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_level(self):
        cb = _cb()
        cb.trigger(3, "테스트")
        cb.reset()
        assert cb.level == 0

    def test_reset_clears_buy_block(self):
        cb = _cb()
        cb.trigger(3, "테스트")
        cb.reset()
        assert cb.is_buy_blocked() is False

    def test_reset_clears_api_error_count(self):
        cb = _cb()
        cb.record_api_error()
        cb.record_api_error()
        cb.reset()
        assert cb.get_status()["api_error_count"] == 0


# ---------------------------------------------------------------------------
# maybe_recover()
# ---------------------------------------------------------------------------

class TestMaybeRecover:
    def test_level0_returns_false(self):
        cb = _cb()
        assert cb.maybe_recover() is False

    def test_level5_no_auto_recover(self):
        cb = _cb()
        cb.trigger(5, "수동 필요")
        # 쿨다운 시간이 아무리 지나도 auto_recover=False
        cb._triggered_at = datetime.now(timezone.utc) - timedelta(days=999)
        assert cb.maybe_recover() is False
        assert cb.level == 5

    def test_level1_auto_recovers_after_cooldown(self):
        cb = _cb()
        cb.trigger(1, "테스트")
        # 쿨다운(5분)을 이미 지난 것으로 조작
        cb._triggered_at = datetime.now(timezone.utc) - timedelta(minutes=6)
        recovered = cb.maybe_recover()
        assert recovered is True
        assert cb.level == 0

    def test_level1_no_recover_before_cooldown(self):
        cb = _cb()
        cb.trigger(1, "테스트")
        # 쿨다운 미경과
        recovered = cb.maybe_recover()
        assert recovered is False
        assert cb.level == 1


# ---------------------------------------------------------------------------
# record_api_error() / clear_api_error()
# ---------------------------------------------------------------------------

class TestApiError:
    def test_single_error_no_trigger(self):
        cb = _cb()
        cb.record_api_error()
        assert cb.level == 0

    def test_two_errors_no_trigger(self):
        cb = _cb()
        cb.record_api_error()
        cb.record_api_error()
        assert cb.level == 0

    def test_three_errors_trigger_level4(self):
        cb = _cb()
        cb.record_api_error()
        cb.record_api_error()
        cb.record_api_error()
        assert cb.level == 4

    def test_clear_api_error_resets_count(self):
        cb = _cb()
        cb.record_api_error()
        cb.record_api_error()
        cb.clear_api_error()
        cb.record_api_error()
        # 카운터 초기화 후 1회 → Level 0
        assert cb.level == 0
        assert cb.get_status()["api_error_count"] == 1


# ---------------------------------------------------------------------------
# check_price_drop()
# ---------------------------------------------------------------------------

class TestCheckPriceDrop:
    def test_no_drop_no_trigger(self):
        cb = _cb()
        result = cb.check_price_drop(
            coin="KRW-BTC",
            price_1m_ago=50_000_000.0,
            price_10m_ago=50_000_000.0,
            current_price=50_000_000.0,
            daily_loss_pct=0.0,
            cumulative_loss_24h=0.0,
        )
        assert result == 0
        assert cb.level == 0

    def test_1m_drop_3pct_triggers_level1(self):
        cb = _cb()
        result = cb.check_price_drop(
            coin="KRW-BTC",
            price_1m_ago=50_000_000.0,
            price_10m_ago=50_000_000.0,
            current_price=48_500_000.0,  # -3%
            daily_loss_pct=0.0,
            cumulative_loss_24h=0.0,
        )
        assert result >= 1
        assert cb.level >= 1

    def test_10m_drop_8pct_triggers_level2(self):
        cb = _cb()
        result = cb.check_price_drop(
            coin="KRW-BTC",
            price_1m_ago=50_000_000.0,
            price_10m_ago=50_000_000.0,
            current_price=46_000_000.0,  # -8%
            daily_loss_pct=0.0,
            cumulative_loss_24h=0.0,
        )
        assert result >= 2
        assert cb.level >= 2

    def test_daily_loss_10pct_triggers_level3(self):
        cb = _cb()
        result = cb.check_price_drop(
            coin="KRW-BTC",
            price_1m_ago=50_000_000.0,
            price_10m_ago=50_000_000.0,
            current_price=50_000_000.0,
            daily_loss_pct=-0.10,
            cumulative_loss_24h=0.0,
        )
        assert result >= 3
        assert cb.level >= 3

    def test_24h_loss_15pct_triggers_level5(self):
        cb = _cb()
        result = cb.check_price_drop(
            coin="KRW-BTC",
            price_1m_ago=50_000_000.0,
            price_10m_ago=50_000_000.0,
            current_price=50_000_000.0,
            daily_loss_pct=0.0,
            cumulative_loss_24h=-0.15,
        )
        assert result == 5
        assert cb.level == 5

    def test_zero_price_no_crash(self):
        """price_1m_ago = 0 → ZeroDivisionError 없어야 함."""
        cb = _cb()
        result = cb.check_price_drop(
            coin="KRW-BTC",
            price_1m_ago=0.0,
            price_10m_ago=0.0,
            current_price=50_000_000.0,
            daily_loss_pct=0.0,
            cumulative_loss_24h=0.0,
        )
        assert result == 0


# ---------------------------------------------------------------------------
# check_data_mismatch()
# ---------------------------------------------------------------------------

class TestCheckDataMismatch:
    def test_below_threshold_no_trigger(self):
        cb = _cb()
        cb.check_data_mismatch(2.9)
        assert cb.level == 0

    def test_above_threshold_triggers_level4(self):
        cb = _cb()
        cb.check_data_mismatch(3.1)
        assert cb.level == 4


# ---------------------------------------------------------------------------
# thread-safety: 여러 스레드에서 동시 trigger
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_triggers_no_exception(self):
        import threading
        cb = _cb()
        errors = []

        def _trigger(lvl: int) -> None:
            try:
                cb.trigger(lvl, f"스레드{lvl}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_trigger, args=(i,)) for i in range(1, 6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert 1 <= cb.level <= 5

    def test_concurrent_record_api_error(self):
        import threading
        cb = _cb()

        def _record() -> None:
            cb.record_api_error()

        threads = [threading.Thread(target=_record) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 3회 이상 → Level 4 발동 보장
        assert cb.level == 4
