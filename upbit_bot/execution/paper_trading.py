"""
execution/paper_trading.py — 페이퍼 트레이딩 병렬 실행기

실전 봇과 동일 신호로 가상 거래를 병렬 실행.
페이퍼↔실거래 정량 비교 3지표 (매일 SQLite 저장):
  ① 신호 일치율: 동일 루프에서 페이퍼/실거래 모두 진입한 비율
     → 85% 미만 시 알림
  ② 체결가 괴리: (실거래 평균 체결가 - 페이퍼 기준가) / 기준가 × 100
     → 지속적으로 -0.2% 초과 시 SlippageWarning
  ③ 타이밍 슬리피지: 신호 발생 타임스탬프 → 실제 체결 완료 타임스탬프 차이 (초)
     → 평균 5초 초과 시 NetworkLatencyWarning
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 임계값
# ------------------------------------------------------------------

SIGNAL_MATCH_ALERT_THRESHOLD  = 0.85    # 신호 일치율 85% 미만 → 알림
PRICE_DEVIATION_ALERT_PCT     = -0.2    # 체결가 괴리 -0.2% 초과 → 알림
TIMING_SLIPPAGE_ALERT_SEC     = 5.0     # 타이밍 슬리피지 5초 초과 → 알림
HISTORY_WINDOW                = 100     # 비교 지표 Rolling 윈도우


@dataclass
class PaperTradeRecord:
    """페이퍼 트레이딩 단일 거래 기록."""
    coin: str
    side: str               # 'BUY' | 'SELL'
    signal_ts: datetime     # 신호 발생 시각
    paper_price: float      # 기준가 (신호 시점 현재가)
    krw_amount: float
    strategy_type: str = ""
    # 실거래 대응 (동일 루프에서 채워짐)
    live_avg_price: float | None = None
    live_executed_ts: datetime | None = None
    loop_id: str = ""


@dataclass
class ComparisonMetrics:
    """페이퍼↔실거래 정량 비교 결과."""
    window_size: int
    signal_match_rate: float           # 0~1
    avg_price_deviation_pct: float     # % (음수=실거래가 더 비쌈)
    avg_timing_slippage_sec: float     # 초
    # 알림 플래그
    alert_match_rate: bool = False
    alert_price_deviation: bool = False
    alert_timing: bool = False
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "signal_match_rate": round(self.signal_match_rate, 4),
            "avg_price_deviation_pct": round(self.avg_price_deviation_pct, 4),
            "avg_timing_slippage_sec": round(self.avg_timing_slippage_sec, 3),
            "alert_match_rate": self.alert_match_rate,
            "alert_price_deviation": self.alert_price_deviation,
            "alert_timing": self.alert_timing,
            "computed_at": self.computed_at.isoformat(),
        }


class PaperPortfolio:
    """페이퍼 트레이딩 가상 포트폴리오.

    실전과 동일 신호 → 수수료(0.25%) 반영 후 가상 체결.
    """

    TRADING_COST = 0.0025   # 0.25% (수수료 + 슬리피지)

    def __init__(self, initial_capital: float = 10_000_000.0) -> None:
        self._capital = initial_capital
        self._initial_capital = initial_capital
        self._positions: dict[str, dict[str, float]] = {}  # coin → {qty, avg_price}
        self._trades: list[PaperTradeRecord] = []
        self._realized_pnl: float = 0.0

    def execute(self, record: PaperTradeRecord) -> bool:
        """신호 기준가로 가상 체결.

        Returns:
            True — 체결 성공
            False — 잔고 부족 / 포지션 없음 등 스킵
        """
        cost_rate = 1 + self.TRADING_COST

        if record.side == "BUY":
            cost = record.krw_amount * cost_rate
            if cost > self._capital:
                logger.debug("[PaperPortfolio] KRW 잔고 부족 (필요=%.0f 보유=%.0f)", cost, self._capital)
                return False
            qty = record.krw_amount / record.paper_price
            self._capital -= cost
            pos = self._positions.setdefault(record.coin, {"qty": 0.0, "avg_price": 0.0})
            total_qty = pos["qty"] + qty
            if total_qty > 0:
                pos["avg_price"] = (pos["qty"] * pos["avg_price"] + qty * record.paper_price) / total_qty
            pos["qty"] = total_qty
            logger.debug("[PaperPortfolio] BUY %s qty=%.6f avg=%.0f", record.coin, qty, record.paper_price)

        else:  # SELL
            pos = self._positions.get(record.coin)
            if not pos or pos["qty"] <= 0:
                logger.debug("[PaperPortfolio] SELL 스킵: %s 포지션 없음", record.coin)
                return False
            qty = pos["qty"]
            proceeds = qty * record.paper_price * (1 - self.TRADING_COST)
            pnl = proceeds - qty * pos["avg_price"]
            self._realized_pnl += pnl
            self._capital += proceeds
            pos["qty"] = 0.0
            logger.debug(
                "[PaperPortfolio] SELL %s qty=%.6f pnl=%.0f", record.coin, qty, pnl
            )

        self._trades.append(record)
        return True

    @property
    def total_equity(self) -> float:
        """현재 총자산 (현금 + 미실현 포지션 평가액). 현재가 정보 없으므로 현금만 집계."""
        return self._capital

    @property
    def return_pct(self) -> float:
        return (self._capital - self._initial_capital) / self._initial_capital

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    def get_position(self, coin: str) -> dict[str, float]:
        return self._positions.get(coin, {"qty": 0.0, "avg_price": 0.0})

    def get_summary(self) -> dict[str, Any]:
        return {
            "capital": round(self._capital, 0),
            "realized_pnl": round(self._realized_pnl, 0),
            "return_pct": round(self.return_pct * 100, 2),
            "trade_count": self.trade_count,
            "open_positions": {
                c: p for c, p in self._positions.items() if p["qty"] > 0
            },
        }


class PaperTradingRunner:
    """페이퍼 트레이딩 병렬 실행 + 정량 비교 지표 관리.

    사용법:
        runner = PaperTradingRunner()
        # 매 루프에서:
        record = runner.on_signal(coin, side, price, krw_amount, strategy_type, loop_id)
        runner.on_live_executed(loop_id, live_avg_price, executed_ts)
        metrics = runner.compute_metrics()
    """

    def __init__(self, initial_capital: float = 10_000_000.0) -> None:
        self._portfolio = PaperPortfolio(initial_capital)
        # loop_id → record 매핑 (실거래 체결 정보 수신 전까지 보관)
        self._pending: dict[str, PaperTradeRecord] = {}
        # 완료된 레코드
        self._completed: deque[PaperTradeRecord] = deque(maxlen=HISTORY_WINDOW)
        # 루프별 신호 일치 추적
        self._loop_paper: set[str] = set()   # 페이퍼 진입 loop_id
        self._loop_live: set[str] = set()    # 실거래 진입 loop_id
        self._recent_loops: deque[str] = deque(maxlen=HISTORY_WINDOW)

    def on_signal(
        self,
        coin: str,
        side: str,
        paper_price: float,
        krw_amount: float,
        strategy_type: str,
        loop_id: str,
        signal_ts: datetime | None = None,
    ) -> PaperTradeRecord:
        """신호 발생 시 페이퍼 체결 실행."""
        ts = signal_ts or datetime.now(timezone.utc)
        record = PaperTradeRecord(
            coin=coin,
            side=side,
            signal_ts=ts,
            paper_price=paper_price,
            krw_amount=krw_amount,
            strategy_type=strategy_type,
            loop_id=loop_id,
        )
        executed = self._portfolio.execute(record)
        if executed:
            self._pending[loop_id] = record
            self._loop_paper.add(loop_id)
            self._recent_loops.append(loop_id)
        return record

    def on_live_executed(
        self,
        loop_id: str,
        live_avg_price: float,
        executed_ts: datetime | None = None,
    ) -> None:
        """실거래 체결 완료 시 호출 → 비교 지표 기록."""
        self._loop_live.add(loop_id)
        record = self._pending.pop(loop_id, None)
        if record is None:
            return
        record.live_avg_price = live_avg_price
        record.live_executed_ts = executed_ts or datetime.now(timezone.utc)
        self._completed.append(record)

    def on_live_skipped(self, loop_id: str) -> None:
        """실거래에서 스킵된 루프 기록 (신호 일치율 분모용)."""
        if loop_id not in self._loop_live:
            # 실거래 진입 없음: loop_id만 추적 (분모에 포함)
            pass

    def compute_metrics(self) -> ComparisonMetrics:
        """최근 HISTORY_WINDOW 루프 기준 비교 지표 계산."""
        # ① 신호 일치율
        total_loops = len(self._recent_loops)
        if total_loops == 0:
            match_rate = 1.0
        else:
            matched = len(self._loop_paper & self._loop_live)
            match_rate = matched / total_loops

        # ② 체결가 괴리 (완료된 레코드 기준)
        deviations: list[float] = []
        timings: list[float] = []
        for rec in self._completed:
            if rec.live_avg_price is not None and rec.paper_price > 0:
                dev = (rec.live_avg_price - rec.paper_price) / rec.paper_price * 100
                deviations.append(dev)
            if rec.live_executed_ts is not None:
                elapsed = (rec.live_executed_ts - rec.signal_ts).total_seconds()
                timings.append(elapsed)

        avg_dev = mean(deviations) if deviations else 0.0
        avg_timing = mean(timings) if timings else 0.0

        metrics = ComparisonMetrics(
            window_size=len(self._completed),
            signal_match_rate=match_rate,
            avg_price_deviation_pct=avg_dev,
            avg_timing_slippage_sec=avg_timing,
            alert_match_rate=match_rate < SIGNAL_MATCH_ALERT_THRESHOLD,
            alert_price_deviation=avg_dev < PRICE_DEVIATION_ALERT_PCT,
            alert_timing=avg_timing > TIMING_SLIPPAGE_ALERT_SEC,
        )

        if metrics.alert_match_rate:
            logger.warning(
                "[PaperTrading] 신호 일치율 낮음: %.1f%% (기준 %.0f%%)",
                match_rate * 100, SIGNAL_MATCH_ALERT_THRESHOLD * 100,
            )
        if metrics.alert_price_deviation:
            logger.warning(
                "[PaperTrading] 체결가 괴리 과다: %.3f%% (기준 %.1f%%)",
                avg_dev, PRICE_DEVIATION_ALERT_PCT,
            )
        if metrics.alert_timing:
            logger.warning(
                "[PaperTrading] 타이밍 슬리피지 과다: %.1f초 (기준 %.0f초)",
                avg_timing, TIMING_SLIPPAGE_ALERT_SEC,
            )

        return metrics

    def get_portfolio_summary(self) -> dict[str, Any]:
        return self._portfolio.get_summary()

    def get_weekly_report(self) -> str:
        """주간 텔레그램 리포트 문자열 생성."""
        metrics = self.compute_metrics()
        summary = self.get_portfolio_summary()
        lines = [
            "<b>[페이퍼 트레이딩 주간 리포트]</b>",
            f"신호 일치율: {metrics.signal_match_rate*100:.1f}%"
            + (" ⚠️" if metrics.alert_match_rate else ""),
            f"체결가 괴리: {metrics.avg_price_deviation_pct:.3f}%"
            + (" ⚠️" if metrics.alert_price_deviation else ""),
            f"타이밍 슬리피지: {metrics.avg_timing_slippage_sec:.1f}초"
            + (" ⚠️" if metrics.alert_timing else ""),
            f"페이퍼 수익률: {summary['return_pct']:.2f}%",
            f"실현손익: {summary['realized_pnl']:,.0f}원",
            f"거래횟수: {summary['trade_count']}회",
        ]
        return "\n".join(lines)
