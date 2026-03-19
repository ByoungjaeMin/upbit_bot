"""
strategies/decay_monitor.py — StrategyDecayMonitor

전략별 4주 롤링 샤프비율 감시 + 동적 자본 배분 가중치 계산.
APScheduler: 매주 일요일 05:00 UTC 전체 전략 성과 재계산.

DORMANT 트리거: 샤프 < 0.5 로 4주 연속
DORMANT 동작: 해당 전략 자본 배분 0% + 텔레그램 알림
복귀 조건: 2주 연속 샤프 > 0.8 후 백테스트 재검증

SQLite strategy_decay_log 저장 → StrategySelector.get_weights() 연동
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DORMANT_SHARPE_THRESHOLD = 0.5    # 이하 4주 → DORMANT
REVIVAL_SHARPE_THRESHOLD = 0.8    # 이상 2주 → 복귀 후보
DORMANT_WEEKS = 4
REVIVAL_WEEKS = 2
STRATEGY_TYPES = ("TREND_STRONG", "TREND_NORMAL", "GRID", "DCA")


class StrategyDecayMonitor:
    """전략 성과 감시 + 동적 가중치 계산기.

    사용법:
        monitor = StrategyDecayMonitor(cache=cache)
        # 매주 일요일 05:00 호출
        monitor.update_weekly_stats(strategy_log_rows)
        weights = monitor.get_weights()
        dormant = monitor.check_dormant()
    """

    def __init__(self, cache: Any = None) -> None:
        self._cache = cache
        # {strategy_type: deque[sharpe]} — 주별 샤프비율 이력 (최대 8주)
        self._sharpe_history: dict[str, list[float]] = {
            s: [] for s in STRATEGY_TYPES
        }
        self._dormant_status: dict[str, bool] = {
            s: False for s in STRATEGY_TYPES
        }
        self._dormant_since: dict[str, datetime | None] = {
            s: None for s in STRATEGY_TYPES
        }

    # ------------------------------------------------------------------
    # 주간 통계 업데이트 (APScheduler 매주 일요일 05:00)
    # ------------------------------------------------------------------

    def update_weekly_stats(
        self,
        strategy_log_rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """전략별 주간 샤프/승률/손익비 계산 → strategy_decay_log 저장.

        Args:
            strategy_log_rows: strategy_log 테이블 rows (dict 리스트)
                각 row: {strategy_type, pnl_pct, pnl, timestamp}

        Returns:
            {strategy_type: {rolling_sharpe, win_rate, profit_loss_ratio, trade_count}}
        """
        from collections import defaultdict

        stats: dict[str, dict[str, float]] = {}
        grouped: dict[str, list[float]] = defaultdict(list)

        for row in strategy_log_rows:
            stype = row.get("strategy_type", "")
            pnl_pct = row.get("pnl_pct")
            if stype in STRATEGY_TYPES and pnl_pct is not None:
                grouped[stype].append(float(pnl_pct))

        week_start = self._current_week_start()

        for stype in STRATEGY_TYPES:
            rets = np.array(grouped[stype], dtype=float)
            trade_count = len(rets)

            if trade_count < 2:
                sharpe = 0.0
                win_rate = 0.0
                pl_ratio = 0.0
            else:
                std = float(rets.std())
                sharpe = float(rets.mean()) / std * math.sqrt(52) if std > 1e-9 else 0.0
                win_rate = float((rets > 0).mean())
                wins = rets[rets > 0]
                losses = rets[rets <= 0]
                avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
                avg_loss = abs(float(losses.mean())) if len(losses) > 0 else 1e-9
                pl_ratio = avg_win / avg_loss if avg_loss > 1e-9 else 0.0

            # 이력 갱신 (최대 8주)
            hist = self._sharpe_history[stype]
            hist.append(sharpe)
            if len(hist) > 8:
                hist.pop(0)

            stats[stype] = {
                "rolling_sharpe": round(sharpe, 4),
                "win_rate": round(win_rate, 4),
                "profit_loss_ratio": round(pl_ratio, 4),
                "trade_count": trade_count,
            }

            self._save_decay_log(stype, week_start, stats[stype])

        return stats

    # ------------------------------------------------------------------
    # 동적 가중치
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, float]:
        """4주 롤링 샤프비율 기반 정규화된 자본 배분 가중치.

        w_i = max(sharpe_i, 0) / Σ max(sharpe_j, 0)
        음수 샤프 전략 → 0 배분.
        모든 전략 0 → 빈 dict 반환 (StrategySelector → HOLD 폴백).
        """
        raw: dict[str, float] = {}
        for stype in STRATEGY_TYPES:
            if self._dormant_status[stype]:
                raw[stype] = 0.0
                continue
            hist = self._sharpe_history[stype]
            if not hist:
                raw[stype] = 0.0
                continue
            # 최근 4주 평균
            recent = hist[-DORMANT_WEEKS:]
            raw[stype] = max(float(np.mean(recent)), 0.0)

        total = sum(raw.values())
        if total < 1e-9:
            return {}

        return {s: round(v / total, 4) for s, v in raw.items()}

    # ------------------------------------------------------------------
    # DORMANT 감지
    # ------------------------------------------------------------------

    def check_dormant(self) -> list[str]:
        """샤프 < 0.5 로 4주 연속인 전략 → DORMANT 전환.

        Returns:
            새로 DORMANT 전환된 전략 목록
        """
        newly_dormant: list[str] = []

        for stype in STRATEGY_TYPES:
            if self._dormant_status[stype]:
                continue
            hist = self._sharpe_history[stype]
            if len(hist) < DORMANT_WEEKS:
                continue
            recent = hist[-DORMANT_WEEKS:]
            if all(s < DORMANT_SHARPE_THRESHOLD for s in recent):
                self._dormant_status[stype] = True
                self._dormant_since[stype] = datetime.now(timezone.utc)
                newly_dormant.append(stype)
                logger.warning(
                    "[DecayMonitor] %s DORMANT 전환 (4주 샤프 < %.1f)",
                    stype, DORMANT_SHARPE_THRESHOLD,
                )

        return newly_dormant

    # ------------------------------------------------------------------
    # 복귀 후보 감지
    # ------------------------------------------------------------------

    def check_revival(self) -> list[str]:
        """DORMANT 전략 중 2주 연속 샤프 > 0.8 → 복귀 후보.

        Returns:
            복귀 후보 전략 목록 (자동 활성화는 하지 않음 — 수동 확인 권고)
        """
        revival_candidates: list[str] = []

        for stype in STRATEGY_TYPES:
            if not self._dormant_status[stype]:
                continue
            hist = self._sharpe_history[stype]
            if len(hist) < REVIVAL_WEEKS:
                continue
            recent = hist[-REVIVAL_WEEKS:]
            if all(s > REVIVAL_SHARPE_THRESHOLD for s in recent):
                revival_candidates.append(stype)
                logger.info(
                    "[DecayMonitor] %s 복귀 후보 (2주 샤프 > %.1f)",
                    stype, REVIVAL_SHARPE_THRESHOLD,
                )

        return revival_candidates

    def revive(self, strategy_type: str) -> None:
        """전략 수동 재활성화 (백테스트 재검증 통과 후 호출)."""
        if strategy_type not in STRATEGY_TYPES:
            raise ValueError(f"알 수 없는 전략: {strategy_type}")
        self._dormant_status[strategy_type] = False
        self._dormant_since[strategy_type] = None
        logger.info("[DecayMonitor] %s 재활성화", strategy_type)

    # ------------------------------------------------------------------
    # 상태 리포트 (/decay 명령어 응답용)
    # ------------------------------------------------------------------

    def get_status_report(self) -> str:
        """전략별 상태 요약 (텔레그램 /decay 응답용)."""
        lines = ["<b>📊 전략 성과 모니터</b>\n"]
        weights = self.get_weights()

        for stype in STRATEGY_TYPES:
            hist = self._sharpe_history[stype]
            sharpe = round(float(np.mean(hist[-4:])), 3) if hist else 0.0
            dormant_flag = "🔴 DORMANT" if self._dormant_status[stype] else "🟢 활성"
            weight_pct = round(weights.get(stype, 0.0) * 100, 1)

            lines.append(
                f"<b>{stype}</b>: {dormant_flag} | 샤프={sharpe} | 배분={weight_pct}%"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Optuna 재탐색 트리거 (DORMANT 진입 시)
    # ------------------------------------------------------------------

    def trigger_reoptimize(
        self,
        strategy_type: str,
        X: Any = None,
        y: Any = None,
        n_trials: int = 50,
    ) -> dict[str, Any]:
        """DORMANT 진입 전략 파라미터 재탐색 (Optuna n_trials=50).

        실제 최적화는 backtest/walk_forward.py와 연동 예정.
        현재는 best_params 구조만 반환 (Phase C 구현).
        """
        logger.info("[DecayMonitor] %s Optuna 재탐색 시작 (n_trials=%d)", strategy_type, n_trials)

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("[DecayMonitor] Optuna 미설치 — 재탐색 불가")
            return {}

        # 실제 Walk-Forward 목적함수는 Phase C에서 구현
        # 여기서는 기본 파라미터 구조 반환
        default_params = {
            "TREND_STRONG": {"adx_min": 30, "take_profit": 0.15, "stop_loss": 0.10},
            "TREND_NORMAL": {"adx_min": 20, "take_profit": 0.08, "stop_loss": 0.07},
            "GRID":  {"atr_multiplier": 3.0, "levels": 10},
            "DCA":   {"step_pct": -0.03, "max_safety": 5, "take_profit_pct": 0.03},
        }
        return default_params.get(strategy_type, {})

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    @staticmethod
    def _current_week_start() -> str:
        """이번 주 월요일 00:00 UTC ISO8601."""
        now = datetime.now(timezone.utc)
        monday = now - __import__("datetime").timedelta(days=now.weekday())
        return monday.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    def _save_decay_log(
        self,
        strategy_type: str,
        week_start: str,
        stat: dict[str, float],
    ) -> None:
        if not self._cache:
            return
        try:
            row = {
                "week_start": week_start,
                "strategy_type": strategy_type,
                "rolling_sharpe": stat["rolling_sharpe"],
                "win_rate": stat["win_rate"],
                "profit_loss_ratio": stat["profit_loss_ratio"],
                "trade_count": int(stat["trade_count"]),
                "is_dormant": int(self._dormant_status[strategy_type]),
                "dormant_since": (
                    self._dormant_since[strategy_type].isoformat()
                    if self._dormant_since[strategy_type]
                    else None
                ),
            }
            self._cache.insert_row("strategy_decay_log", row)
        except Exception as exc:
            logger.error("[DecayMonitor] DB 저장 실패: %s", exc)
