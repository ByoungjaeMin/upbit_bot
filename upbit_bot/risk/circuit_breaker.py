"""
risk/circuit_breaker.py — 5단계 서킷브레이커

모든 레이어보다 최우선 (CLAUDE.md 원칙 3번).
Layer 1 필터의 조건 0에서 반드시 최우선 체크.

Level 0: 정상 운영
Level 1: 1분 내 -3%  → 매수 5분 정지 + 트레일링 스탑 타이트
Level 2: 10분 내 -8% → 전량 USDT 전환 + 30분 중단
Level 3: 일일 -10%   → 당일 전체 중단
Level 4: API 오류 3회 OR WebSocket vs REST 괴리 >3% → 즉시 중단
Level 5: 24h -15%   → 수동 확인 요청 (자동 재개 없음)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 레벨별 설정
# ------------------------------------------------------------------

_LEVEL_CONFIG = {
    1: {"block_buy": True,  "block_sell": False, "cooldown_min": 5,   "auto_recover": True},
    2: {"block_buy": True,  "block_sell": False, "cooldown_min": 30,  "auto_recover": True},
    3: {"block_buy": True,  "block_sell": False, "cooldown_min": 1440,"auto_recover": True},
    4: {"block_buy": True,  "block_sell": False, "cooldown_min": 60,  "auto_recover": True},
    5: {"block_buy": True,  "block_sell": False, "cooldown_min": 99999,"auto_recover": False},
}


class TriggerEvent(NamedTuple):
    level: int
    reason: str
    timestamp: datetime


class CircuitBreaker:
    """5단계 서킷브레이커 (thread-safe).

    사용법:
        cb = CircuitBreaker()

        # 체크
        if cb.is_buy_blocked():
            return  # 매수 차단

        # 트리거
        cb.trigger(1, "1분 -3% 감지")

        # 자동 회복 체크 (메인 루프 10초 주기)
        cb.maybe_recover()
    """

    def __init__(self, cache=None) -> None:
        self._cache = cache
        self._level: int = 0
        self._triggered_at: datetime | None = None
        self._reason: str = ""
        self._history: list[TriggerEvent] = []
        self._api_error_count: int = 0
        self._lock = Lock()

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------

    @property
    def level(self) -> int:
        with self._lock:
            return self._level

    def is_buy_blocked(self) -> bool:
        """매수 차단 여부 (Level 1~5)."""
        with self._lock:
            return self._level >= 1

    def is_all_blocked(self) -> bool:
        """전체 거래 중단 여부 (Level 2~5)."""
        with self._lock:
            return self._level >= 2

    def is_manual_required(self) -> bool:
        """수동 확인 필요 (Level 5)."""
        with self._lock:
            return self._level == 5

    def get_status(self) -> dict:
        with self._lock:
            return {
                "level": self._level,
                "reason": self._reason,
                "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
                "api_error_count": self._api_error_count,
            }

    # ------------------------------------------------------------------
    # 트리거 / 회복
    # ------------------------------------------------------------------

    def trigger(self, level: int, reason: str) -> None:
        """서킷브레이커 발동 (기존보다 높은 레벨만 적용)."""
        if level < 1 or level > 5:
            raise ValueError(f"유효하지 않은 레벨: {level}")
        with self._lock:
            if level > self._level:
                self._level = level
                self._triggered_at = datetime.now(timezone.utc)
                self._reason = reason
                event = TriggerEvent(level, reason, self._triggered_at)
                self._history.append(event)
                logger.warning(
                    "[서킷브레이커] Level %d 발동: %s", level, reason
                )

    def maybe_recover(self) -> bool:
        """쿨다운 완료 시 자동 회복. True 반환 시 회복됨."""
        with self._lock:
            if self._level == 0:
                return False
            cfg = _LEVEL_CONFIG.get(self._level, {})
            if not cfg.get("auto_recover", True):
                return False
            if self._triggered_at is None:
                return False
            elapsed = datetime.now(timezone.utc) - self._triggered_at
            cooldown = timedelta(minutes=cfg["cooldown_min"])
            if elapsed >= cooldown:
                old = self._level
                self._level = 0
                self._triggered_at = None
                self._reason = ""
                logger.info("[서킷브레이커] Level %d → 0 자동 회복", old)
                return True
        return False

    def reset(self) -> None:
        """강제 초기화 (관리자 수동 명령 전용)."""
        with self._lock:
            self._level = 0
            self._triggered_at = None
            self._reason = ""
            self._api_error_count = 0
        logger.info("[서킷브레이커] 강제 초기화")

    # ------------------------------------------------------------------
    # API 오류 카운터 (Level 4 트리거용)
    # ------------------------------------------------------------------

    def record_api_error(self) -> None:
        """API 오류 1회 기록. 3회 누적 시 Level 4 자동 발동."""
        with self._lock:
            self._api_error_count += 1
            count = self._api_error_count
        if count >= 3:
            self.trigger(4, f"API 오류 {count}회 연속")

    def clear_api_error(self) -> None:
        """API 성공 시 카운터 초기화."""
        with self._lock:
            self._api_error_count = 0

    # ------------------------------------------------------------------
    # 가격 변화 체크 (메인 루프 연동)
    # ------------------------------------------------------------------

    def check_price_drop(
        self,
        coin: str,
        price_1m_ago: float,
        price_10m_ago: float,
        current_price: float,
        daily_loss_pct: float,
        cumulative_loss_24h: float,
    ) -> int:
        """가격 기반 서킷브레이커 조건 체크 후 해당 레벨 트리거.

        Returns:
            발동된 레벨 (0 = 미발동)
        """
        triggered = 0

        # Level 1: 1분 내 -3%
        if price_1m_ago > 0:
            drop_1m = (current_price - price_1m_ago) / price_1m_ago
            if drop_1m <= -0.03:
                self.trigger(1, f"[{coin}] 1분 내 {drop_1m:.1%} 급락")
                triggered = max(triggered, 1)

        # Level 2: 10분 내 -8%
        if price_10m_ago > 0:
            drop_10m = (current_price - price_10m_ago) / price_10m_ago
            if drop_10m <= -0.08:
                self.trigger(2, f"[{coin}] 10분 내 {drop_10m:.1%} 급락")
                triggered = max(triggered, 2)

        # Level 3: 일일 -10%
        if daily_loss_pct <= -0.10:
            self.trigger(3, f"일일 손실 {daily_loss_pct:.1%}")
            triggered = max(triggered, 3)

        # Level 5: 24h -15%
        if cumulative_loss_24h <= -0.15:
            self.trigger(5, f"24h 누적 손실 {cumulative_loss_24h:.1%} — 수동 확인 필요")
            triggered = max(triggered, 5)

        return triggered

    def check_data_mismatch(self, deviation_pct: float) -> None:
        """WebSocket vs REST 괴리 >3% → Level 4."""
        if deviation_pct > 3.0:
            self.trigger(4, f"WebSocket vs REST 괴리 {deviation_pct:.2f}% > 3%")

    def get_history(self, last_n: int = 10) -> list[TriggerEvent]:
        with self._lock:
            return list(self._history[-last_n:])
