"""test_telegram_bot.py — TelegramBot + MessageFormatter 단위 테스트."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# python-telegram-bot 미설치 환경에서도 테스트 가능하도록
sys.modules.setdefault("telegram", MagicMock())
sys.modules.setdefault("telegram.constants", MagicMock())
sys.modules.setdefault("telegram.ext", MagicMock())

from upbit_bot.monitoring.telegram_bot import (  # noqa: E402
    BotContext,
    MessageFormatter,
    TelegramBot,
    create_bot_from_env,
)


def run(coro):  # type: ignore[no-untyped-def]
    """pytest-asyncio 없이 async 테스트 실행 헬퍼."""
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────────────
# MessageFormatter 테스트
# ──────────────────────────────────────────────────────────────────────

class TestMessageFormatter:
    def setup_method(self) -> None:
        self.fmt = MessageFormatter()

    def test_buy_format_contains_coin(self) -> None:
        msg = self.fmt.buy(
            coin="KRW-BTC",
            price=100_000_000,
            amount_krw=50_000,
            capital_pct=2.0,
            ensemble_prob=0.65,
            consensus=3,
            strategy="TREND_STRONG",
            position_n=1,
        )
        assert "KRW-BTC" in msg
        assert "매수체결" in msg
        assert "0.650" in msg
        assert "TREND_STRONG" in msg

    def test_sell_profit_emoji_green(self) -> None:
        msg = self.fmt.sell(
            coin="KRW-ETH",
            entry_price=3_000_000,
            exit_price=3_300_000,
            pnl_pct=0.10,
            pnl_krw=300_000,
            reason="익절",
        )
        assert "🟢" in msg
        assert "+10.00%" in msg

    def test_sell_loss_emoji_red(self) -> None:
        msg = self.fmt.sell(
            coin="KRW-ETH",
            entry_price=3_000_000,
            exit_price=2_700_000,
            pnl_pct=-0.10,
            pnl_krw=-300_000,
            reason="손절",
        )
        assert "🔴" in msg
        assert "-10.00%" in msg

    def test_grid_fill_format(self) -> None:
        msg = self.fmt.grid_fill(
            coin="KRW-SOL",
            level=3,
            side="매수",
            cumulative_profit_krw=12_000,
        )
        assert "그리드체결" in msg
        assert "KRW-SOL" in msg
        assert "레벨: 3" in msg

    def test_dca_order_pnl(self) -> None:
        msg = self.fmt.dca_order(
            coin="KRW-XRP",
            safety_n=2,
            avg_price=1_000,
            current_price=900,
            total_invested=50_000,
        )
        assert "DCA매수" in msg
        assert "-10.00%" in msg

    def test_dca_order_zero_avg_price(self) -> None:
        """avg_price=0 일 때 ZeroDivisionError 없어야 함."""
        msg = self.fmt.dca_order(
            coin="KRW-XRP",
            safety_n=1,
            avg_price=0,
            current_price=900,
            total_invested=10_000,
        )
        assert "DCA매수" in msg

    def test_circuit_breaker_format(self) -> None:
        msg = self.fmt.circuit_breaker(
            level=2,
            reason="-8% 급락",
            action="USDT 전환",
            resume_at="30분 후",
        )
        assert "서킷브레이커 Level 2" in msg
        assert "-8% 급락" in msg

    def test_retrain_format(self) -> None:
        msg = self.fmt.retrain(accuracy=0.72, wf_sharpe=1.45)
        assert "72.0%" in msg
        assert "1.450" in msg

    def test_daily_report_format(self) -> None:
        msg = self.fmt.daily_report(
            total_trades=5,
            realized_pnl=80_000,
            win_rate=0.6,
            capital=1_000_000,
            strategy_contrib={"TREND": 1.5, "GRID": 0.3},
            paper_match_rate=0.85,
            disk_free_gb=120.5,
        )
        assert "일간리포트" in msg
        assert "TREND" in msg
        assert "85.0%" in msg
        assert "120.5GB" in msg

    def test_error_format(self) -> None:
        msg = self.fmt.error("API 연결 실패")
        assert "오류" in msg
        assert "API 연결 실패" in msg


# ──────────────────────────────────────────────────────────────────────
# BotContext 테스트
# ──────────────────────────────────────────────────────────────────────

class TestBotContext:
    def test_defaults(self) -> None:
        ctx = BotContext()
        assert ctx.stop_new_buys is False
        assert ctx.emergency_pending is False
        assert ctx.current_phase == "Phase A"
        assert ctx.engine is None


# ──────────────────────────────────────────────────────────────────────
# TelegramBot 초기화 테스트
# ──────────────────────────────────────────────────────────────────────

class TestTelegramBotInit:
    def test_init_with_explicit_values(self) -> None:
        bot = TelegramBot(token="tok", chat_id=12345)
        assert bot._token == "tok"
        assert bot._chat_id == 12345

    def test_init_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_TOKEN", "env_tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "99999")
        bot = TelegramBot()
        assert bot._token == "env_tok"
        assert bot._chat_id == 99999

    def test_set_context(self) -> None:
        bot = TelegramBot(token="t", chat_id=1)
        ctx = BotContext(current_phase="Phase B")
        bot.set_context(ctx)
        assert bot._ctx.current_phase == "Phase B"


# ──────────────────────────────────────────────────────────────────────
# 화이트리스트 테스트
# ──────────────────────────────────────────────────────────────────────

class TestTelegramBotWhitelist:
    def _make_update(self, chat_id: int) -> MagicMock:
        update = MagicMock()
        update.effective_chat.id = chat_id
        return update

    def test_whitelist_match(self) -> None:
        bot = TelegramBot(token="t", chat_id=42)
        assert bot._check_whitelist(self._make_update(42)) is True

    def test_whitelist_mismatch(self) -> None:
        bot = TelegramBot(token="t", chat_id=42)
        assert bot._check_whitelist(self._make_update(99)) is False

    def test_whitelist_none_update(self) -> None:
        bot = TelegramBot(token="t", chat_id=42)
        assert bot._check_whitelist(None) is False


# ──────────────────────────────────────────────────────────────────────
# 큐 메시지 전송 테스트
# ──────────────────────────────────────────────────────────────────────

class TestTelegramBotQueue:
    def test_send_enqueues_message(self) -> None:
        bot = TelegramBot(token="t", chat_id=1)

        async def _run() -> None:
            await bot.send("hello", priority=1)
            item = bot._queue.get_nowait()
            assert item["text"] == "hello"
            assert item["priority"] == 1

        run(_run())

    def test_send_buy_helper(self) -> None:
        bot = TelegramBot(token="t", chat_id=1)

        async def _run() -> None:
            await bot.send_buy(
                coin="KRW-BTC",
                price=100_000_000,
                amount_krw=50_000,
                capital_pct=2.0,
                ensemble_prob=0.65,
                consensus=3,
                strategy="TREND_STRONG",
                position_n=1,
            )
            item = bot._queue.get_nowait()
            assert "매수체결" in item["text"]
            assert item["priority"] == 1

        run(_run())

    def test_send_sell_helper(self) -> None:
        bot = TelegramBot(token="t", chat_id=1)

        async def _run() -> None:
            await bot.send_sell(
                coin="KRW-ETH",
                entry_price=3_000_000,
                exit_price=3_300_000,
                pnl_pct=0.1,
                pnl_krw=300_000,
                reason="익절",
            )
            item = bot._queue.get_nowait()
            assert "매도체결" in item["text"]

        run(_run())

    def test_send_circuit_helper(self) -> None:
        bot = TelegramBot(token="t", chat_id=1)

        async def _run() -> None:
            await bot.send_circuit(
                level=3, reason="일일 -10%", action="당일 중단", resume_at="내일"
            )
            item = bot._queue.get_nowait()
            assert "서킷브레이커" in item["text"]
            assert item["priority"] == 1

        run(_run())


# ──────────────────────────────────────────────────────────────────────
# 명령어 핸들러 테스트
# ──────────────────────────────────────────────────────────────────────

class TestTelegramBotCommands:
    def _make_bot(self) -> TelegramBot:
        return TelegramBot(token="t", chat_id=42)

    def _make_update(self, chat_id: int = 42) -> MagicMock:
        update = MagicMock()
        update.effective_chat.id = chat_id
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()
        return update

    def test_guard_blocks_unknown_user(self) -> None:
        bot = self._make_bot()
        update = self._make_update(chat_id=999)

        async def _run() -> None:
            result = await bot._guard(update, MagicMock())
            assert result is False
            # silent drop — 미등록 사용자에게 응답하지 않음
            update.message.reply_text.assert_not_awaited()

        run(_run())

    def test_guard_allows_owner(self) -> None:
        bot = self._make_bot()
        update = self._make_update(chat_id=42)

        async def _run() -> None:
            result = await bot._guard(update, MagicMock())
            assert result is True

        run(_run())

    def test_cmd_stop_sets_flag(self) -> None:
        bot = self._make_bot()
        update = self._make_update()

        async def _run() -> None:
            await bot._cmd_stop(update, MagicMock())
            assert bot._ctx.stop_new_buys is True
            update.message.reply_text.assert_awaited_once()

        run(_run())

    def test_cmd_phase_returns_phase(self) -> None:
        bot = self._make_bot()
        bot._ctx.current_phase = "Phase B"
        update = self._make_update()

        async def _run() -> None:
            await bot._cmd_phase(update, MagicMock())
            call_text = update.message.reply_text.call_args[0][0]
            assert "Phase B" in call_text

        run(_run())

    def test_cmd_status_no_engine(self) -> None:
        bot = self._make_bot()
        update = self._make_update()

        async def _run() -> None:
            await bot._cmd_status(update, MagicMock())
            update.message.reply_text.assert_awaited_once()

        run(_run())

    def test_cmd_balance_no_engine(self) -> None:
        bot = self._make_bot()
        update = self._make_update()

        async def _run() -> None:
            await bot._cmd_balance(update, MagicMock())
            update.message.reply_text.assert_awaited_once()

        run(_run())

    def test_cmd_emergency_sets_pending(self) -> None:
        bot = self._make_bot()
        update = self._make_update()

        async def _run() -> None:
            with patch(
                "upbit_bot.monitoring.telegram_bot.asyncio.sleep",
                new_callable=AsyncMock,
            ):
                await bot._cmd_emergency(update, MagicMock())
            update.message.reply_text.assert_awaited()

        run(_run())

    def test_cmd_confirm_without_emergency(self) -> None:
        bot = self._make_bot()
        bot._ctx.emergency_pending = False
        update = self._make_update()

        async def _run() -> None:
            await bot._cmd_confirm(update, MagicMock())
            update.message.reply_text.assert_awaited_once()

        run(_run())

    def test_cmd_confirm_with_emergency(self) -> None:
        bot = self._make_bot()
        bot._ctx.emergency_pending = True
        update = self._make_update()

        mock_engine = MagicMock()
        mock_engine.emergency_liquidate_all = AsyncMock(return_value={"KRW-BTC": True})
        bot._ctx.engine = mock_engine
        bot._app = MagicMock()
        bot._app.bot.send_message = AsyncMock()

        async def _run() -> None:
            await bot._cmd_confirm(update, MagicMock())
            assert bot._ctx.emergency_pending is False
            assert bot._ctx.stop_new_buys is True
            mock_engine.emergency_liquidate_all.assert_awaited_once()

        run(_run())

    def test_cmd_confirm_raises_when_no_engine(self) -> None:
        bot = self._make_bot()
        bot._ctx.emergency_pending = True
        bot._ctx.engine = None
        update = self._make_update()

        async def _run() -> None:
            with pytest.raises(RuntimeError, match="engine 미연결"):
                await bot._cmd_confirm(update, MagicMock())

        run(_run())

    def test_cmd_mode_no_args(self) -> None:
        bot = self._make_bot()
        update = self._make_update()
        ctx = MagicMock()
        ctx.args = []

        async def _run() -> None:
            await bot._cmd_mode(update, ctx)
            update.message.reply_text.assert_awaited_once()

        run(_run())

    def test_cmd_mode_dry(self) -> None:
        bot = self._make_bot()
        engine = MagicMock()
        engine._dry_run = False
        bot._ctx.engine = engine
        update = self._make_update()
        ctx = MagicMock()
        ctx.args = ["dry"]

        async def _run() -> None:
            await bot._cmd_mode(update, ctx)
            assert engine._dry_run is True

        run(_run())

    def test_cmd_mode_live_under_200(self) -> None:
        bot = self._make_bot()
        engine = MagicMock()
        engine.trade_count = 50
        bot._ctx.engine = engine
        update = self._make_update()
        ctx = MagicMock()
        ctx.args = ["live"]

        async def _run() -> None:
            await bot._cmd_mode(update, ctx)
            call_text = update.message.reply_text.call_args[0][0]
            assert "DRY_RUN" in call_text

        run(_run())

    def test_cmd_decay_no_monitor(self) -> None:
        bot = self._make_bot()
        update = self._make_update()

        async def _run() -> None:
            await bot._cmd_decay(update, MagicMock())
            update.message.reply_text.assert_awaited_once()

        run(_run())

    def test_cmd_paper_with_runner(self) -> None:
        bot = self._make_bot()
        runner = MagicMock()
        runner.get_weekly_report.return_value = "📊 페이퍼 리포트"
        bot._ctx.paper_runner = runner
        update = self._make_update()

        async def _run() -> None:
            await bot._cmd_paper(update, MagicMock())
            call_text = update.message.reply_text.call_args[0][0]
            assert "페이퍼" in call_text

        run(_run())


# ──────────────────────────────────────────────────────────────────────
# 재시도 로직 테스트
# ──────────────────────────────────────────────────────────────────────

class TestSendWithRetry:
    def test_send_success_on_first_try(self) -> None:
        bot = TelegramBot(token="t", chat_id=1)
        mock_app = MagicMock()
        mock_app.bot.send_message = AsyncMock()
        bot._app = mock_app

        async def _run() -> None:
            await bot._send_with_retry("hello")
            mock_app.bot.send_message.assert_awaited_once()

        run(_run())

    def test_send_retries_on_failure(self) -> None:
        bot = TelegramBot(token="t", chat_id=1)
        mock_app = MagicMock()
        mock_app.bot.send_message = AsyncMock(
            side_effect=[Exception("err"), Exception("err"), None]
        )
        bot._app = mock_app

        async def _run() -> None:
            with patch(
                "upbit_bot.monitoring.telegram_bot.asyncio.sleep",
                new_callable=AsyncMock,
            ):
                await bot._send_with_retry("hello")
            assert mock_app.bot.send_message.await_count == 3

        run(_run())

    def test_send_no_app(self) -> None:
        """_app이 None일 때 예외 없이 반환해야 함."""
        bot = TelegramBot(token="t", chat_id=1)

        async def _run() -> None:
            await bot._send_with_retry("hello")  # 예외 없음

        run(_run())


# ──────────────────────────────────────────────────────────────────────
# 팩토리 함수 테스트
# ──────────────────────────────────────────────────────────────────────

class TestCreateBotFromEnv:
    def test_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_TOKEN", "factory_tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "777")
        bot = create_bot_from_env()
        assert bot._token == "factory_tok"
        assert bot._chat_id == 777

    def test_factory_with_queue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_TOKEN", "tok2")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "888")
        q: asyncio.Queue = asyncio.Queue()
        bot = create_bot_from_env(event_queue=q)
        assert bot._queue is q
