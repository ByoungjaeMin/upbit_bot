"""
utils/helpers.py — 공통 유틸리티 함수 모음

카테고리:
  - 시간: utc_now(), kst_now(), floor_to_5m(), ts_to_str()
  - 수치: clamp(), pct_change(), safe_div()
  - 코인: strip_market_prefix(), is_krw_market()
  - DB: rows_to_df()
  - 포맷: fmt_krw(), fmt_pct()
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# 시간 유틸리티
# ---------------------------------------------------------------------------

_KST = timezone(timedelta(hours=9))


def utc_now() -> datetime:
    """현재 UTC datetime (tzinfo 포함)."""
    return datetime.now(timezone.utc)


def kst_now() -> datetime:
    """현재 KST datetime (UTC+9)."""
    return datetime.now(_KST)


def floor_to_5m(dt: datetime) -> datetime:
    """datetime을 5분 단위로 내림 (초·마이크로초 제거)."""
    return dt.replace(second=0, microsecond=0, minute=(dt.minute // 5) * 5)


def ts_to_str(dt: datetime) -> str:
    """datetime → ISO 8601 UTC 문자열 (SQLite 저장용)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def str_to_ts(s: str) -> datetime:
    """ISO 8601 문자열 → datetime (tzinfo=UTC)."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# 수치 유틸리티
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    """value를 [lo, hi] 범위로 제한."""
    return max(lo, min(hi, value))


def pct_change(current: float, previous: float) -> float:
    """변화율 (%) 계산. previous=0 이면 0.0 반환."""
    if previous == 0:
        return 0.0
    return (current - previous) / abs(previous) * 100.0


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """ZeroDivisionError 방지 나눗셈."""
    if denominator == 0:
        return default
    return numerator / denominator


def round_price(price: float, tick: float = 1.0) -> float:
    """업비트 호가단위(tick)로 반올림."""
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 10)


# ---------------------------------------------------------------------------
# 코인 유틸리티
# ---------------------------------------------------------------------------

def strip_market_prefix(coin: str) -> str:
    """'KRW-BTC' → 'BTC'."""
    return coin.split("-")[-1]


def is_krw_market(coin: str) -> bool:
    """KRW 마켓 여부 확인."""
    return coin.upper().startswith("KRW-")


# ---------------------------------------------------------------------------
# DB 유틸리티
# ---------------------------------------------------------------------------

def rows_to_df(
    rows: Sequence[sqlite3.Row] | list[dict[str, Any]],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """sqlite3.Row 리스트 → DataFrame 변환."""
    if not rows:
        return pd.DataFrame(columns=columns or [])
    if isinstance(rows[0], sqlite3.Row):
        data = [dict(r) for r in rows]
    else:
        data = list(rows)
    df = pd.DataFrame(data)
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns]
    return df


# ---------------------------------------------------------------------------
# 포맷 유틸리티
# ---------------------------------------------------------------------------

def fmt_krw(amount: float, sign: bool = False) -> str:
    """KRW 금액 포맷 (예: +1,234,567원)."""
    prefix = "+" if (sign and amount >= 0) else ""
    return f"{prefix}{amount:,.0f}원"


def fmt_pct(ratio: float, decimals: int = 2, sign: bool = False) -> str:
    """비율 포맷 (0.0~1.0 → %, 예: +12.34%)."""
    prefix = "+" if (sign and ratio >= 0) else ""
    return f"{prefix}{ratio * 100:.{decimals}f}%"
