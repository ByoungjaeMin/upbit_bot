"""
monitoring/telegram_bot.py — 텔레그램 봇 (python-telegram-bot 20.x async)

[알림 메시지 포맷]
  매수:   🟢 매수체결 / 코인,가격,금액(자본%) / 앙상블확률,합의수 / 전략타입 / 포지션N/3
  매도:   🔴 매도체결 / 매수가→매도가 / 손익%,원 / 사유
  그리드: ⚡ 그리드체결 / 코인,레벨 / 누적수익
  DCA:    💧 DCA매수 / Safety Order N / 평균단가,현재가격
  서킷:   🚨 서킷브레이커 Level N / 사유 / 조치 / 재개예정
  재학습: 🔄 앙상블재학습완료 / 정확도 / Walk-Forward샤프
  일간리포트(밤11시): 📊

[전체 명령어 25개]
  /status /balance /scan /strategy /stop /emergency /report
  /layer1 /ensemble /hmm /kelly /grid /dca /retrain /hyperopt
  /quality /storage /vacuum /cleanup /paper /mode /phase
  /decay /kimchi /pairs /montecarlo

[보안]
  - .env: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
  - 화이트리스트: CHAT_ID 일치하는 사용자만 명령어 수신
  - /emergency → /confirm 이중확인 필요
  - 자동 재연결 + 3회 재시도
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    from telegram import (
        Bot,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Update,
    )
    from telegram.constants import ParseMode
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
    )
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False
    logger.warning("[TelegramBot] python-telegram-bot 미설치 — 봇 비활성화")

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

MAX_RETRY           = 3
RECONNECT_DELAY_SEC = 5
EMERGENCY_TOKEN     = "CONFIRM_EMERGENCY"   # /emergency → /confirm 이중확인 키


# ------------------------------------------------------------------
# 메시지 포맷터
# ------------------------------------------------------------------

class MessageFormatter:
    """텔레그램 알림 메시지 HTML 포맷터."""

    @staticmethod
    def buy(
        coin: str,
        price: float,
        amount_krw: float,
        capital_pct: float,
        ensemble_prob: float,
        consensus: int,
        strategy: str,
        position_n: int,
        position_max: int = 5,
    ) -> str:
        return (
            f"🟢 <b>매수체결</b>\n"
            f"코인: <b>{coin}</b> | 가격: {price:,.0f}원\n"
            f"금액: {amount_krw:,.0f}원 (<b>{capital_pct:.1f}%</b>)\n"
            f"앙상블: {ensemble_prob:.3f} | 합의: {consensus}모델\n"
            f"전략: <b>{strategy}</b> | 포지션: {position_n}/{position_max}"
        )

    @staticmethod
    def sell(
        coin: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        pnl_krw: float,
        reason: str,
    ) -> str:
        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        sign  = "+" if pnl_pct >= 0 else ""
        return (
            f"{emoji} <b>매도체결</b>\n"
            f"코인: <b>{coin}</b>\n"
            f"진입가: {entry_price:,.0f} → 청산가: {exit_price:,.0f}원\n"
            f"손익: <b>{sign}{pnl_pct*100:.2f}%</b> ({sign}{pnl_krw:,.0f}원)\n"
            f"사유: {reason}"
        )

    @staticmethod
    def grid_fill(
        coin: str,
        level: int,
        side: str,
        cumulative_profit_krw: float,
    ) -> str:
        return (
            f"⚡ <b>그리드체결</b>\n"
            f"코인: <b>{coin}</b> | 레벨: {level} ({side})\n"
            f"누적수익: {cumulative_profit_krw:+,.0f}원"
        )

    @staticmethod
    def dca_order(
        coin: str,
        safety_n: int,
        avg_price: float,
        current_price: float,
        total_invested: float,
    ) -> str:
        pnl_pct = (current_price - avg_price) / avg_price * 100 if avg_price else 0
        return (
            f"💧 <b>DCA매수</b>\n"
            f"코인: <b>{coin}</b> | Safety Order {safety_n}\n"
            f"평균단가: {avg_price:,.0f}원 | 현재가: {current_price:,.0f}원\n"
            f"미실현: {pnl_pct:+.2f}% | 투입총액: {total_invested:,.0f}원"
        )

    @staticmethod
    def circuit_breaker(
        level: int,
        reason: str,
        action: str,
        resume_at: str,
    ) -> str:
        return (
            f"🚨 <b>서킷브레이커 Level {level}</b>\n"
            f"사유: {reason}\n"
            f"조치: {action}\n"
            f"재개예정: {resume_at}"
        )

    @staticmethod
    def retrain(
        accuracy: float,
        wf_sharpe: float,
        model_type: str = "앙상블",
    ) -> str:
        return (
            f"🔄 <b>{model_type} 재학습 완료</b>\n"
            f"정확도: {accuracy:.1%} | Walk-Forward 샤프: {wf_sharpe:.3f}"
        )

    @staticmethod
    def daily_report(
        total_trades: int,
        realized_pnl: float,
        win_rate: float,
        capital: float,
        strategy_contrib: dict[str, float],
        paper_match_rate: float,
        disk_free_gb: float,
    ) -> str:
        strat_lines = "\n".join(
            f"  {k}: {v:+.2f}%" for k, v in strategy_contrib.items()
        )
        return (
            f"📊 <b>일간리포트</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
            f"총거래: {total_trades}회 | 실현손익: {realized_pnl:+,.0f}원\n"
            f"승률: {win_rate:.1%} | 자본: {capital:,.0f}원\n"
            f"<b>전략별 기여도:</b>\n{strat_lines}\n"
            f"페이퍼 신호일치: {paper_match_rate:.1%}\n"
            f"디스크잔여: {disk_free_gb:.1f}GB"
        )

    @staticmethod
    def error(message: str) -> str:
        return f"⚠️ <b>오류</b>\n{message}"


# ------------------------------------------------------------------
# BotContext — 엔진 상태 참조 컨테이너
# ------------------------------------------------------------------

@dataclass
class BotContext:
    """텔레그램 봇이 참조하는 엔진 상태 컨테이너.

    engine / paper_runner 등을 직접 주입하여 /status, /balance 등 명령어가
    실시간 상태를 조회할 수 있게 한다.
    """
    engine: Any = None                  # TradingEngine
    paper_runner: Any = None            # PaperTradingRunner
    circuit_breaker: Any = None         # CircuitBreaker
    decay_monitor: Any = None           # StrategyDecayMonitor
    layer1: Any = None
    layer2: Any = None
    db_conn: Any = None                 # sqlite3.Connection (대시보드 조회용)
    stop_new_buys: bool = False
    emergency_pending: bool = False     # /emergency 이중확인 대기
    current_phase: str = "Phase A"


# ------------------------------------------------------------------
# TelegramBot
# ------------------------------------------------------------------

class TelegramBot:
    """python-telegram-bot 20.x 기반 비동기 텔레그램 봇.

    퀀트 엔진과 같은 asyncio 루프에서 실행.
    asyncio.Queue를 통해 엔진 → 봇 메시지 전달.

    사용법:
        bot = TelegramBot(token=TOKEN, chat_id=CHAT_ID)
        bot.set_context(ctx)
        await bot.start()         # 봇 실행 (블로킹)
        await bot.send(msg)       # 외부에서 메시지 큐잉
    """

    def __init__(
        self,
        token: str = "",
        chat_id: int | str = 0,
        event_queue: asyncio.Queue | None = None,
    ) -> None:
        resolved_token = token or os.getenv("TELEGRAM_TOKEN", "")
        if not resolved_token:
            raise ValueError(
                "TELEGRAM_TOKEN 미설정 — .env 또는 환경변수에 TELEGRAM_TOKEN을 추가하라."
            )
        resolved_chat_id = int(chat_id or os.getenv("TELEGRAM_CHAT_ID", "0"))
        if resolved_chat_id == 0:
            raise ValueError(
                "TELEGRAM_CHAT_ID 미설정 또는 0 — .env 또는 환경변수에 TELEGRAM_CHAT_ID를 추가하라."
            )
        self._token   = resolved_token
        self._chat_id = resolved_chat_id
        self._queue: asyncio.Queue[dict[str, Any]] = event_queue or asyncio.Queue(maxsize=100)
        self._ctx = BotContext()
        self._app: Any = None
        self._formatter = MessageFormatter()
        self._shutdown_event: asyncio.Event | None = None  # start()에서 주입

    def set_context(self, ctx: BotContext) -> None:
        self._ctx = ctx

    # ------------------------------------------------------------------
    # 봇 시작 / 종료
    # ------------------------------------------------------------------

    async def start(self, shutdown_event: asyncio.Event | None = None) -> None:
        """봇 애플리케이션 시작 (폴링 + 큐 전송 루프 병렬).

        shutdown_event 가 set 되면 폴링을 중단하고 정상 반환한다.
        """
        if not _TELEGRAM_AVAILABLE:
            logger.error("[TelegramBot] python-telegram-bot 미설치 — 시작 불가")
            return

        if not self._token:
            logger.error("[TelegramBot] TELEGRAM_TOKEN 미설정")
            return

        self._app = (
            Application.builder()
            .token(self._token)
            .build()
        )
        self._register_handlers()

        self._shutdown_event = shutdown_event  # 루프에서 종료 신호 체크용
        logger.info("[TelegramBot] 시작 chat_id=%d", self._chat_id)
        async with self._app:
            await self._app.start()
            polling_task = asyncio.create_task(
                self._app.updater.start_polling(drop_pending_updates=True)
            )
            dispatch_task = asyncio.create_task(self._queue_dispatch_loop())

            if shutdown_event:
                await shutdown_event.wait()
                # dispatch 루프 먼저 취소
                dispatch_task.cancel()
                await asyncio.gather(dispatch_task, return_exceptions=True)
                # PTB 정식 종료 순서: updater → app (async with __aexit__ 전에 호출 필수)
                if self._app.updater.running:
                    await self._app.updater.stop()
                if self._app.running:
                    await self._app.stop()
                await asyncio.gather(polling_task, return_exceptions=True)
            else:
                await asyncio.gather(polling_task, dispatch_task)

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            logger.info("[TelegramBot] 종료")

    # ------------------------------------------------------------------
    # 외부 메시지 큐잉 (엔진 → 봇)
    # ------------------------------------------------------------------

    async def send(
        self,
        message: str,
        priority: int = 2,
        parse_mode: str = "HTML",
    ) -> None:
        """메시지를 큐에 추가. 우선순위 1=즉시 / 2=일반 / 3=배치."""
        try:
            self._queue.put_nowait({
                "text": message,
                "priority": priority,
                "parse_mode": parse_mode,
            })
        except asyncio.QueueFull:
            logger.warning("[TelegramBot] 메시지 큐 포화(maxsize=100) — 드롭: %.80s", message)

    async def send_buy(self, **kwargs) -> None:
        msg = self._formatter.buy(**kwargs)
        await self.send(msg, priority=1)

    async def send_sell(self, **kwargs) -> None:
        msg = self._formatter.sell(**kwargs)
        await self.send(msg, priority=1)

    async def send_circuit(self, **kwargs) -> None:
        msg = self._formatter.circuit_breaker(**kwargs)
        await self.send(msg, priority=1)

    async def send_daily_report(self, **kwargs) -> None:
        msg = self._formatter.daily_report(**kwargs)
        await self.send(msg, priority=2)

    async def send_grid(self, **kwargs) -> None:
        await self.send(self._formatter.grid_fill(**kwargs), priority=2)

    async def send_dca(self, **kwargs) -> None:
        await self.send(self._formatter.dca_order(**kwargs), priority=2)

    async def send_retrain(self, **kwargs) -> None:
        await self.send(self._formatter.retrain(**kwargs), priority=2)

    # ------------------------------------------------------------------
    # 큐 디스패치 루프
    # ------------------------------------------------------------------

    async def _queue_dispatch_loop(self) -> None:
        """큐에서 메시지를 꺼내 텔레그램으로 전송. 재시도 3회.

        종료 신호: start()에서 dispatch_task.cancel() 호출 → CancelledError → break.
        self._shutdown_event 직접 체크는 _queue.get() 대기 중에도 응답 가능.
        """
        while True:
            # 종료 신호 체크 — task.cancel() 없이도 shutdown_event로 루프 탈출
            if self._shutdown_event is not None and self._shutdown_event.is_set():
                break
            try:
                item = await self._queue.get()
                await self._send_with_retry(
                    item["text"],
                    item.get("parse_mode", "HTML"),
                )
                self._queue.task_done()
            except asyncio.CancelledError:
                break  # start()의 dispatch_task.cancel() → 정상 종료
            except Exception as exc:
                logger.error("[TelegramBot] 큐 디스패치 오류: %s", exc)

    async def _send_with_retry(self, text: str, parse_mode: str = "HTML") -> None:
        """3회 재시도 + 지수 백오프."""
        if not self._app:
            return
        for attempt in range(1, MAX_RETRY + 1):
            try:
                await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode=parse_mode,
                )
                return
            except Exception as exc:
                logger.warning("[TelegramBot] 전송 시도 %d 실패: %s", attempt, exc)
                if attempt < MAX_RETRY:
                    await asyncio.sleep(RECONNECT_DELAY_SEC * attempt)
        logger.error("[TelegramBot] 메시지 전송 최종 실패: %s", text[:80])

    # ------------------------------------------------------------------
    # 화이트리스트 가드
    # ------------------------------------------------------------------

    def _check_whitelist(self, update: Any) -> bool:
        """CHAT_ID 화이트리스트 확인."""
        if not update or not update.effective_chat:
            return False
        return int(update.effective_chat.id) == self._chat_id

    # ------------------------------------------------------------------
    # 명령어 핸들러 등록
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        if not self._app:
            return
        cmds = [
            ("start",      self._cmd_start),
            ("status",     self._cmd_status),
            ("balance",    self._cmd_balance),
            ("scan",       self._cmd_scan),
            ("strategy",   self._cmd_strategy),
            ("stop",       self._cmd_stop),
            ("emergency",  self._cmd_emergency),
            ("confirm",    self._cmd_confirm),
            ("report",     self._cmd_report),
            ("layer1",     self._cmd_layer1),
            ("ensemble",   self._cmd_ensemble),
            ("hmm",        self._cmd_hmm),
            ("kelly",      self._cmd_kelly),
            ("grid",       self._cmd_grid),
            ("dca",        self._cmd_dca),
            ("retrain",    self._cmd_retrain),
            ("hyperopt",   self._cmd_hyperopt),
            ("quality",    self._cmd_quality),
            ("storage",    self._cmd_storage),
            ("vacuum",     self._cmd_vacuum),
            ("cleanup",    self._cmd_cleanup),
            ("paper",      self._cmd_paper),
            ("mode",       self._cmd_mode),
            ("phase",      self._cmd_phase),
            ("decay",      self._cmd_decay),
            ("kimchi",     self._cmd_kimchi),
            ("pairs",      self._cmd_pairs),
            ("montecarlo", self._cmd_montecarlo),
        ]
        for name, handler in cmds:
            self._app.add_handler(CommandHandler(name, handler))
        self._app.add_handler(CallbackQueryHandler(self._callback_query))

    # ------------------------------------------------------------------
    # 명령어 구현
    # ------------------------------------------------------------------

    async def _guard(self, update: Any, context: Any) -> bool:
        """화이트리스트 가드 — 미등록 사용자 차단."""
        if not self._check_whitelist(update):
            chat_id = (
                update.effective_chat.id
                if update and update.effective_chat else "unknown"
            )
            logger.warning("[TelegramBot] 미등록 접근 차단 (command): chat_id=%s", chat_id)
            return False
        return True

    async def _cmd_start(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        keyboard = [
            [InlineKeyboardButton("📊 상태", callback_data="status"),
             InlineKeyboardButton("💰 잔고", callback_data="balance")],
            [InlineKeyboardButton("🔍 스캔", callback_data="scan"),
             InlineKeyboardButton("📈 전략", callback_data="strategy")],
            [InlineKeyboardButton("📋 리포트", callback_data="report"),
             InlineKeyboardButton("🛑 중단", callback_data="stop")],
        ]
        markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "🤖 <b>업비트 퀀트봇</b>\n명령어를 선택하세요:",
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    async def _cmd_status(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        engine = self._ctx.engine
        if engine is None:
            await update.message.reply_text("⚠️ 엔진 미연결")
            return

        status = engine.get_status()
        cb = self._ctx.circuit_breaker
        cb_level = cb.level if cb else 0

        lines = [
            "📊 <b>봇 상태</b>",
            f"DRY_RUN: {'✅' if status.get('dry_run') else '🔴실거래'}",
            f"서킷브레이커: Level {cb_level}",
            f"포지션: {status.get('open_positions', 0)}개",
            f"일거래횟수: {status.get('daily_trade_count', 0)}/{10}",
            f"총거래: {status.get('trade_count', 0)}건",
            f"자본: {status.get('capital', 0):,.0f}원",
            f"신규매수: {'중단' if self._ctx.stop_new_buys else '정상'}",
            f"현재Phase: {self._ctx.current_phase}",
        ]
        positions = engine.positions
        if positions:
            lines.append("\n<b>보유 포지션:</b>")
            for coin, pos in positions.items():
                lines.append(f"  {coin}: {pos.hold_minutes:.0f}분 보유")

        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def _cmd_balance(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        engine = self._ctx.engine
        if engine is None:
            await update.message.reply_text("⚠️ 엔진 미연결")
            return
        status = engine.get_status()
        await update.message.reply_text(
            f"💰 <b>잔고</b>\n자본: {status.get('capital', 0):,.0f}원",
            parse_mode=ParseMode.HTML,
        )

    async def _cmd_scan(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        conn = self._ctx.db_conn
        if conn is None:
            await update.message.reply_text("⚠️ DB 미연결")
            return
        try:
            rows = conn.execute(
                """SELECT coin, rank_by_volume, volume_24h_krw, included
                   FROM coin_scan_results
                   WHERE timestamp = (SELECT MAX(timestamp) FROM coin_scan_results)
                   ORDER BY rank_by_volume"""
            ).fetchall()
            if not rows:
                await update.message.reply_text("📋 스캔 결과 없음 (아직 실행 전)")
                return
            lines = ["🔍 <b>최신 코인 스캔</b>"]
            for r in rows[:15]:
                mark = "✅" if r["included"] else "❌"
                vol = r["volume_24h_krw"] / 1e8 if r["volume_24h_krw"] else 0
                lines.append(f"{mark} #{r['rank_by_volume']} {r['coin']} ({vol:.1f}억)")
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
        except Exception as exc:
            await update.message.reply_text(f"⚠️ 오류: {exc}")

    async def _cmd_strategy(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        conn = self._ctx.db_conn
        if conn is None:
            await update.message.reply_text("⚠️ DB 미연결")
            return
        try:
            rows = conn.execute(
                """SELECT strategy_type, COUNT(*) as cnt,
                          SUM(pnl) as total_pnl,
                          AVG(CASE WHEN pnl>0 THEN 1.0 ELSE 0.0 END) as win_rate
                   FROM trades
                   WHERE side='SELL' AND is_dry_run=0
                   GROUP BY strategy_type
                   ORDER BY total_pnl DESC"""
            ).fetchall()
            lines = ["📈 <b>전략별 성과</b>"]
            for r in rows:
                lines.append(
                    f"• {r['strategy_type']}: {r['cnt']}건 "
                    f"손익 {r['total_pnl']:+,.0f}원 "
                    f"승률 {r['win_rate']:.0%}"
                )
            if not rows:
                lines.append("거래 기록 없음")
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
        except Exception as exc:
            await update.message.reply_text(f"⚠️ 오류: {exc}")

    async def _cmd_stop(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        self._ctx.stop_new_buys = True
        await update.message.reply_text("🛑 신규매수 중단. /stop 해제하려면 /mode 사용")

    async def _cmd_emergency(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        if self._ctx.emergency_pending:
            await update.message.reply_text("⚠️ 이미 긴급 대기 중 — /confirm 또는 30초 대기")
            return
        self._ctx.emergency_pending = True
        await update.message.reply_text(
            "🚨 <b>긴급 전량매도</b> 준비\n"
            "30초 내 /confirm 을 입력하면 전량매도+중단 실행\n"
            "취소: 아무것도 입력하지 않으면 자동 취소",
            parse_mode=ParseMode.HTML,
        )
        # 30초 후 자동 취소
        await asyncio.sleep(30)
        if self._ctx.emergency_pending:
            self._ctx.emergency_pending = False
            await self._send_with_retry("⚠️ /emergency 이중확인 시간 초과 — 취소됨")

    async def _cmd_confirm(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        if not self._ctx.emergency_pending:
            await update.message.reply_text("⚠️ 먼저 /emergency 를 입력하세요")
            return
        self._ctx.emergency_pending = False
        self._ctx.stop_new_buys = True
        engine = self._ctx.engine
        if engine is None:
            raise RuntimeError("[TelegramBot] /confirm: engine 미연결 — 긴급 청산 불가")
        logger.warning("[TelegramBot] /confirm 수신 — 긴급 전량매도 트리거")
        results = await engine.emergency_liquidate_all()
        failed = [c for c, ok in results.items() if not ok]
        if failed:
            await self._send_with_retry(
                f"⚠️ 긴급 청산 부분 실패: {failed}\n포지션 루프에서 재시도 예정"
            )
        else:
            await self._send_with_retry(
                f"✅ 긴급 청산 완료: {len(results)}개 포지션 청산됨"
            )

    async def _cmd_report(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        conn = self._ctx.db_conn
        if conn is None:
            await update.message.reply_text("⚠️ DB 미연결")
            return
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            rows = conn.execute(
                """SELECT coin, side, price, krw_amount, pnl, strategy_type
                   FROM trades
                   WHERE timestamp >= ? AND is_dry_run=0
                   ORDER BY timestamp DESC LIMIT 20""",
                (today,),
            ).fetchall()
            summary = conn.execute(
                """SELECT COUNT(*) as cnt, SUM(CASE WHEN side='SELL' THEN pnl ELSE 0 END) as pnl
                   FROM trades WHERE timestamp >= ? AND is_dry_run=0""",
                (today,),
            ).fetchone()
            lines = [f"📋 <b>오늘 거래내역</b> ({today})"]
            lines.append(f"총 {summary['cnt']}건 | 실현손익 {(summary['pnl'] or 0):+,.0f}원")
            for r in rows[:10]:
                emoji = "🟢" if r["side"] == "BUY" else "🔴"
                pnl_str = f" {r['pnl']:+,.0f}원" if r["pnl"] else ""
                lines.append(f"{emoji} {r['coin']} {r['side']} {r['price']:,.0f}원{pnl_str}")
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
        except Exception as exc:
            await update.message.reply_text(f"⚠️ 오류: {exc}")

    async def _cmd_layer1(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        conn = self._ctx.db_conn
        if conn is None:
            await update.message.reply_text("⚠️ DB 미연결")
            return
        try:
            rows = conn.execute(
                """SELECT coin, tradeable, regime_strategy, signal_multiplier,
                          adx_value, active_warnings, timestamp
                   FROM layer1_log
                   WHERE timestamp = (
                       SELECT MAX(l2.timestamp) FROM layer1_log l2 WHERE l2.coin=layer1_log.coin
                   )
                   ORDER BY tradeable DESC, coin"""
            ).fetchall()
            lines = ["🔎 <b>Layer1 최신 상태</b>"]
            for r in rows[:10]:
                mark = "✅" if r["tradeable"] else "❌"
                warns = json.loads(r["active_warnings"] or "[]")
                warn_str = f" ⚠️{len(warns)}" if warns else ""
                lines.append(
                    f"{mark} {r['coin']} [{r['regime_strategy']}]"
                    f" ADX={r['adx_value']:.1f} ×{r['signal_multiplier']:.2f}{warn_str}"
                )
            if not rows:
                lines.append("기록 없음")
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
        except Exception as exc:
            await update.message.reply_text(f"⚠️ 오류: {exc}")

    async def _cmd_ensemble(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        conn = self._ctx.db_conn
        if conn is None:
            await update.message.reply_text("⚠️ DB 미연결")
            return
        try:
            rows = conn.execute(
                """SELECT coin, weighted_avg, consensus_count, signal_confirmed,
                          hmm_regime, timestamp
                   FROM ensemble_predictions
                   WHERE timestamp = (
                       SELECT MAX(e2.timestamp) FROM ensemble_predictions e2
                       WHERE e2.coin = ensemble_predictions.coin
                   )
                   ORDER BY weighted_avg DESC LIMIT 10"""
            ).fetchall()
            lines = ["🧠 <b>앙상블 최신 예측</b>"]
            for r in rows:
                conf = "✅" if r["signal_confirmed"] else "◻️"
                lines.append(
                    f"{conf} {r['coin']} prob={r['weighted_avg']:.3f}"
                    f" 합의={r['consensus_count']}개"
                )
            if not rows:
                lines.append("예측 기록 없음")
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
        except Exception as exc:
            await update.message.reply_text(f"⚠️ 오류: {exc}")

    async def _cmd_hmm(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        await update.message.reply_text("📡 HMM 레짐 — Phase C 이후 활성화")

    async def _cmd_kelly(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        engine = self._ctx.engine
        if engine is None:
            await update.message.reply_text("⚠️ 엔진 미연결")
            return
        sizer = getattr(engine, "_kelly_sizer", None)
        if sizer is None:
            await update.message.reply_text("📐 KellySizer 미연결 (engine._kelly_sizer)")
            return
        status = engine.get_status()
        capital = status.get("capital", 0)
        lines = [
            "📐 <b>Kelly 사이징</b>",
            f"총 자본: {capital:,.0f}원",
            f"최대 단일 포지션: {capital * 0.3:,.0f}원 (30%)",
            f"최소 포지션: 50,000원",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def _cmd_grid(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        engine = self._ctx.engine
        if engine is None:
            await update.message.reply_text("⚠️ 엔진 미연결")
            return
        positions = getattr(engine, "positions", {})
        grid_pos = {k: v for k, v in positions.items()
                    if getattr(v, "strategy_type", "") == "GRID"}
        if not grid_pos:
            await update.message.reply_text("⚡ 활성 그리드 포지션 없음")
            return
        lines = ["⚡ <b>그리드 현황</b>"]
        for coin, pos in grid_pos.items():
            lines.append(f"• {coin}: 진입가 {getattr(pos, 'entry_price', 0):,.0f}원")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def _cmd_dca(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        engine = self._ctx.engine
        if engine is None:
            await update.message.reply_text("⚠️ 엔진 미연결")
            return
        positions = getattr(engine, "positions", {})
        dca_pos = {k: v for k, v in positions.items()
                   if getattr(v, "strategy_type", "") == "DCA"}
        if not dca_pos:
            await update.message.reply_text("💧 활성 DCA 포지션 없음")
            return
        lines = ["💧 <b>DCA 현황</b>"]
        for coin, pos in dca_pos.items():
            so = getattr(pos, "safety_order_count", 0)
            avg = getattr(pos, "avg_price", 0)
            lines.append(f"• {coin}: SO={so} 평균단가 {avg:,.0f}원")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def _cmd_retrain(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        layer2 = self._ctx.layer2
        if layer2 is None:
            await update.message.reply_text("⚠️ Layer2 앙상블 미연결")
            return
        await update.message.reply_text("🔄 재학습 트리거 요청 수신 — 백그라운드 실행 중...")
        try:
            if hasattr(layer2, "trigger_retrain"):
                asyncio.create_task(layer2.trigger_retrain())
            else:
                await update.message.reply_text("⚠️ trigger_retrain() 미구현")
        except Exception as exc:
            await update.message.reply_text(f"⚠️ 재학습 오류: {exc}")

    async def _cmd_hyperopt(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        await update.message.reply_text(
            "⚙️ <b>Hyperopt</b>\n"
            "백테스트 Hyperopt는 별도 실행 필요:\n"
            "<code>python -m upbit_bot.backtest.hyperopt</code>\n"
            "(1~2시간 소요, 봇과 별도 프로세스)",
            parse_mode=ParseMode.HTML,
        )

    async def _cmd_quality(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        conn = self._ctx.db_conn
        if conn is None:
            await update.message.reply_text("⚠️ DB 미연결")
            return
        try:
            row = conn.execute(
                """SELECT COUNT(*) as total,
                          SUM(tradeable) as tradeable_cnt
                   FROM layer1_log
                   WHERE timestamp >= datetime('now', '-1 hour')"""
            ).fetchone()
            total = row["total"] or 0
            tradeable = row["tradeable_cnt"] or 0
            rate = tradeable / total if total else 0
            candle_row = conn.execute(
                "SELECT COUNT(DISTINCT coin) as coins FROM candles_5m"
                " WHERE timestamp >= datetime('now', '-10 minutes')"
            ).fetchone()
            active_coins = candle_row["coins"] if candle_row else 0
            await update.message.reply_text(
                f"✅ <b>데이터 품질 (최근 1시간)</b>\n"
                f"Layer1 체크: {total}건\n"
                f"Tradeable: {tradeable} ({rate:.0%})\n"
                f"활성 코인 (10분): {active_coins}개",
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:
            await update.message.reply_text(f"⚠️ 오류: {exc}")

    async def _cmd_storage(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        conn = self._ctx.db_conn
        if conn is None:
            await update.message.reply_text("⚠️ DB 미연결")
            return
        try:
            import shutil
            import os
            # DB 크기
            db_path = getattr(conn, "path", None)
            if db_path is None:
                # sqlite3 Connection에서 DB 파일 경로 가져오기
                row = conn.execute("PRAGMA database_list").fetchone()
                db_path = row[2] if row else None
            db_mb = (os.path.getsize(db_path) / 1024 / 1024) if db_path and os.path.exists(db_path) else 0
            disk = shutil.disk_usage("/")
            free_gb = disk.free / 1024 ** 3
            # 테이블별 행 수
            tables_row = conn.execute(
                "SELECT SUM(rows_deleted) as del, MAX(timestamp) as last_run"
                " FROM storage_audit_log"
            ).fetchone()
            last_run = tables_row["last_run"] or "없음"
            del_rows = tables_row["del"] or 0
            await update.message.reply_text(
                f"💾 <b>스토리지 현황</b>\n"
                f"DB 크기: {db_mb:.1f} MB\n"
                f"디스크 여유: {free_gb:.1f} GB\n"
                f"최근 정리: {last_run}\n"
                f"삭제된 행: {del_rows:,}건",
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:
            await update.message.reply_text(f"⚠️ 오류: {exc}")

    async def _cmd_vacuum(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        engine = self._ctx.engine
        storage = getattr(engine, "_storage_manager", None) if engine else None
        if storage is None:
            await update.message.reply_text("⚠️ StorageManager 미연결")
            return
        await update.message.reply_text("🗜️ VACUUM 실행 중... (수십 초 소요)")
        if engine is not None:
            storage.set_engine(engine)
        try:
            await asyncio.to_thread(storage.vacuum_database)
            await self._send_with_retry("✅ VACUUM 완료")
        except Exception as exc:
            await self._send_with_retry(f"⚠️ VACUUM 오류: {exc}")

    async def _cmd_cleanup(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        engine = self._ctx.engine
        storage = getattr(engine, "_storage_manager", None) if engine else None
        if storage is None:
            await update.message.reply_text("⚠️ StorageManager 미연결")
            return
        await update.message.reply_text("🧹 전체 정리 실행 중...")
        try:
            await asyncio.to_thread(storage.cleanup_candles)
            await asyncio.to_thread(storage.cleanup_logs)
            await self._send_with_retry("✅ 정리 완료")
        except Exception as exc:
            await self._send_with_retry(f"⚠️ 정리 오류: {exc}")

    async def _cmd_paper(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        runner = self._ctx.paper_runner
        if runner is None:
            await update.message.reply_text("⚠️ 페이퍼 트레이딩 미연결")
            return
        report = runner.get_weekly_report()
        await update.message.reply_text(report, parse_mode=ParseMode.HTML)

    async def _cmd_mode(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        args = context.args if context.args else []
        if not args:
            engine = self._ctx.engine
            mode = "DRY_RUN" if (engine and engine.is_dry_run) else "LIVE"
            await update.message.reply_text(f"현재 모드: {mode}\n변경: /mode dry | /mode live")
            return

        mode = args[0].lower()
        if mode == "dry":
            if self._ctx.engine:
                self._ctx.engine._dry_run = True
            await update.message.reply_text("✅ DRY_RUN 모드로 전환")
        elif mode == "live":
            engine = self._ctx.engine
            if engine and engine.trade_count < 200:
                await update.message.reply_text(
                    f"⚠️ trade_count={engine.trade_count} < 200 — DRY_RUN 유지"
                )
            else:
                await update.message.reply_text("🔴 LIVE 모드 전환 — 실제 주문 실행됩니다")
        else:
            await update.message.reply_text("사용법: /mode dry | /mode live")

    async def _cmd_phase(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        await update.message.reply_text(
            f"현재 구현 Phase: <b>{self._ctx.current_phase}</b>",
            parse_mode=ParseMode.HTML,
        )

    async def _cmd_decay(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        monitor = self._ctx.decay_monitor
        if monitor is None:
            await update.message.reply_text("⚠️ StrategyDecayMonitor 미연결")
            return
        report = monitor.get_status_report()
        await update.message.reply_text(
            f"📉 <b>전략 Decay 현황</b>\n{report}",
            parse_mode=ParseMode.HTML,
        )

    async def _cmd_kimchi(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        conn = self._ctx.db_conn
        if conn is None:
            await update.message.reply_text("⚠️ DB 미연결")
            return
        try:
            rows = conn.execute(
                """SELECT timestamp, upbit_btc_krw, binance_btc_usd,
                          usd_krw_rate, kimchi_premium_pct
                   FROM kimchi_premium_log
                   ORDER BY timestamp DESC LIMIT 5"""
            ).fetchall()
            if not rows:
                await update.message.reply_text("🌶️ 김치프리미엄 기록 없음")
                return
            latest = rows[0]
            history = " | ".join(f"{r['kimchi_premium_pct']:+.2f}%" for r in rows)
            await update.message.reply_text(
                f"🌶️ <b>김치프리미엄</b>\n"
                f"최신: <b>{latest['kimchi_premium_pct']:+.2f}%</b>\n"
                f"업비트 BTC: {latest['upbit_btc_krw']:,.0f}원\n"
                f"바이낸스 BTC: ${latest['binance_btc_usd']:,.0f}\n"
                f"환율: {latest['usd_krw_rate']:,.0f}원/$\n"
                f"최근 5회: {history}",
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:
            await update.message.reply_text(f"⚠️ 오류: {exc}")

    async def _cmd_pairs(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        conn = self._ctx.db_conn
        if conn is None:
            await update.message.reply_text("⚠️ DB 미연결")
            return
        try:
            rows = conn.execute(
                """SELECT coin, rank_by_volume, volume_24h_krw
                   FROM coin_scan_results
                   WHERE timestamp = (SELECT MAX(timestamp) FROM coin_scan_results)
                     AND included = 1
                   ORDER BY rank_by_volume"""
            ).fetchall()
            if not rows:
                await update.message.reply_text("📋 활성 페어리스트 없음")
                return
            lines = [f"📋 <b>활성 페어리스트</b> ({len(rows)}개)"]
            for r in rows:
                vol = (r["volume_24h_krw"] or 0) / 1e8
                lines.append(f"#{r['rank_by_volume']} {r['coin']} ({vol:.0f}억)")
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
        except Exception as exc:
            await update.message.reply_text(f"⚠️ 오류: {exc}")

    async def _cmd_montecarlo(self, update: Any, context: Any) -> None:
        if not await self._guard(update, context):
            return
        await update.message.reply_text(
            "🎲 <b>Monte Carlo 검증</b>\n"
            "백테스트 Monte Carlo는 별도 실행 필요:\n"
            "<code>python -m upbit_bot.backtest.monte_carlo</code>",
            parse_mode=ParseMode.HTML,
        )

    # ------------------------------------------------------------------
    # 인라인 키보드 콜백
    # ------------------------------------------------------------------

    async def _callback_query(self, update: Any, context: Any) -> None:
        query = update.callback_query
        await query.answer()

        if not self._check_whitelist(update):
            chat_id = (
                update.effective_chat.id
                if update and update.effective_chat else "unknown"
            )
            logger.warning("[TelegramBot] 미등록 접근 차단 (callback): chat_id=%s", chat_id)
            return

        dispatch = {
            "status":   self._cmd_status,
            "balance":  self._cmd_balance,
            "scan":     self._cmd_scan,
            "strategy": self._cmd_strategy,
            "report":   self._cmd_report,
            "stop":     self._cmd_stop,
        }
        handler = dispatch.get(query.data)
        if handler:
            # callback_query 에는 message가 없으므로 reply_text 대신 edit_message_text
            try:
                await handler(update, context)
            except Exception as exc:
                logger.warning("[TelegramBot] 콜백 핸들러 오류: %s", exc)


# ------------------------------------------------------------------
# 팩토리 함수
# ------------------------------------------------------------------

def create_bot_from_env(event_queue: asyncio.Queue | None = None) -> TelegramBot:
    """환경변수에서 봇 생성."""
    token   = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "0")
    return TelegramBot(token=token, chat_id=chat_id, event_queue=event_queue)
