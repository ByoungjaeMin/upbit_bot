"""
data/quality.py — DataQualityChecker 7단계 파이프라인

순서 (반드시 유지):
  step1  OHLCV 논리 오류 제거
  step2  타임스탬프 정렬 / 중복 / 갭 보간
  step3  거래량 이상치 플래그 (제거 X)
  step4  가격 이상치 (IQR + Z-score + 순간변화율)
  step5  IsolationForest 이상 감지
  step6  데이터 신선도 (stale_data)
  step7  WebSocket vs REST 교차 검증

점수 기준: ≥0.9 정상 / 0.7~0.9 경고 / 0.5~0.7 학습보류 / <0.5 중단
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from risk.circuit_breaker import CircuitBreaker

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

EXPECTED_FREQ: dict[str, str] = {"5m": "5min", "1h": "h", "1d": "D"}
MAX_LAG: dict[str, timedelta] = {
    "5m": timedelta(minutes=10),
    "1h": timedelta(hours=2),
    "1d": timedelta(hours=26),
}
MAX_GAP_INTERPOLATION = 3   # 연속 갭 보간 허용 최대 개수


# ------------------------------------------------------------------
# QualityReport
# ------------------------------------------------------------------

@dataclass
class QualityReport:
    coin: str
    interval: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ohlcv_errors: int = 0
    duplicate_count: int = 0
    gap_count: int = 0
    gaps_interpolated: int = 0
    gaps_excluded: int = 0
    volume_outlier_count: int = 0
    price_outlier_count: int = 0
    anomaly_count: int = 0
    anomaly_pct: float = 0.0       # 0.0~1.0
    stale_data: bool = False
    lag_seconds: float = 0.0
    source_mismatch: bool = False
    source_warning: bool = False
    price_deviation_pct: float = 0.0
    score: float = 1.0
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return DataQualityChecker.score_to_status(self.score)


# ------------------------------------------------------------------
# DataQualityChecker
# ------------------------------------------------------------------

def _run_isolation_forest(feats_values: "np.ndarray") -> "np.ndarray":
    """IsolationForest fit_predict — run_in_executor 격리용 모듈 레벨 함수.

    step5_anomaly_detection에서 await loop.run_in_executor(None, _run_isolation_forest, feats.values)
    형태로 호출한다. 스레드 풀에서 실행되므로 asyncio 이벤트 루프를 블로킹하지 않는다.
    """
    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    return iso.fit_predict(feats_values)


class DataQualityChecker:
    """7단계 데이터 품질 검증 파이프라인.

    사용법:
        checker = DataQualityChecker()
        df, score, report = checker.validate_pipeline(df, "5m", "KRW-BTC",
                                                      ws_price=95_000_000.0)
    """

    def __init__(
        self,
        cache=None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """
        cache:            CandleCache 인스턴스 (리포트 저장용, 선택적)
        circuit_breaker:  CircuitBreaker 인스턴스 (Level 4 발동용, 선택적)
                          None이면 trigger 호출 스킵 + logger.warning 경고
        """
        self._cache = cache
        self._cb = circuit_breaker

    # ------------------------------------------------------------------
    # 메인 파이프라인
    # ------------------------------------------------------------------

    async def validate_pipeline(
        self,
        df: pd.DataFrame,
        interval: str,
        coin: str,
        ws_price: float | None = None,
    ) -> tuple[pd.DataFrame, float, QualityReport]:
        """7단계 순차 검증 — 순서 반드시 유지."""
        report = QualityReport(coin=coin, interval=interval)

        df, report = self.step1_ohlcv_logic(df, report)
        if df.empty:
            report.score = 0.0
            return df, 0.0, report

        df, report = self.step2_timestamp(df, interval, report)
        df, report = self.step3_volume_outliers(df, report)
        df, report = self.step4_price_outliers(df, report)
        df, report = await self.step5_anomaly_detection(df, report)
        report = self.step6_freshness(df, interval, report)
        if ws_price is not None:
            report = self.step7_cross_validate(df, ws_price, report)

        report.score = self._compute_score(report)
        self._log_report(report)
        return df, report.score, report

    # ------------------------------------------------------------------
    # step1: OHLCV 논리 오류 → 즉시 제거
    # ------------------------------------------------------------------

    def step1_ohlcv_logic(
        self, df: pd.DataFrame, report: QualityReport
    ) -> tuple[pd.DataFrame, QualityReport]:
        before = len(df)
        invalid = (
            (df["high"] < df["low"])
            | (df["close"] > df["high"])
            | (df["close"] < df["low"])
            | (df["open"] <= 0)
            | (df["close"] <= 0)
            | (df["volume"] < 0)
        )
        df = df[~invalid].copy()
        removed = before - len(df)
        report.ohlcv_errors = removed
        if removed:
            report.warnings.append(f"OHLCV 논리 오류 {removed}개 제거")
        return df, report

    # ------------------------------------------------------------------
    # step2: 타임스탬프 정렬 / 중복 / 갭 보간
    # ------------------------------------------------------------------

    def step2_timestamp(
        self, df: pd.DataFrame, interval: str, report: QualityReport
    ) -> tuple[pd.DataFrame, QualityReport]:
        # DatetimeIndex 확보
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # 미래 타임스탬프 제거
        now_utc = pd.Timestamp.now(tz="UTC")
        df = df[df.index <= now_utc]

        # 중복 제거 (최신 유지)
        dup_count = int(df.index.duplicated().sum())
        df = df[~df.index.duplicated(keep="last")]
        report.duplicate_count = dup_count

        # 시간 오름차순 정렬
        df = df.sort_index()

        # 갭 탐지 + 보간
        freq = EXPECTED_FREQ.get(interval)
        if freq and len(df) > 1:
            expected_idx = pd.date_range(
                start=df.index[0], end=df.index[-1], freq=freq, tz="UTC"
            )
            missing_count = len(expected_idx.difference(df.index))
            report.gap_count = missing_count

            if missing_count > 0:
                df = df.reindex(expected_idx)
                nan_mask = df["close"].isna()
                # 연속 NaN 블록 크기 계산
                group = (nan_mask != nan_mask.shift()).cumsum()
                consecutive = nan_mask.groupby(group).transform("sum")
                interpolate_mask = nan_mask & (consecutive <= MAX_GAP_INTERPOLATION)
                exclude_mask = nan_mask & (consecutive > MAX_GAP_INTERPOLATION)

                df = df.interpolate(method="time", limit=MAX_GAP_INTERPOLATION)
                df.loc[exclude_mask, "close"] = np.nan
                df = df.dropna(subset=["close"])

                report.gaps_interpolated = int(interpolate_mask.sum())
                report.gaps_excluded = int(exclude_mask.sum())
                if report.gaps_excluded:
                    report.warnings.append(
                        f"연속 갭 {MAX_GAP_INTERPOLATION}개 초과 — {report.gaps_excluded}개 제외"
                    )

        return df, report

    # ------------------------------------------------------------------
    # step3: 거래량 이상치 플래그 (제거 X — 학습 가중치 조정용)
    # ------------------------------------------------------------------

    def step3_volume_outliers(
        self, df: pd.DataFrame, report: QualityReport
    ) -> tuple[pd.DataFrame, QualityReport]:
        if "volume" not in df.columns or len(df) < 5:
            return df, report

        vol = df["volume"]
        std = vol.std()
        if std > 0:
            z = (vol - vol.mean()) / std
            high_vol = z.abs() > 4
            report.volume_outlier_count = int(high_vol.sum())
            if high_vol.any():
                df = df.copy()
                df["volume_outlier"] = high_vol

        # 0 거래량 연속 3개 이상 → 보간
        zero_vol = vol == 0
        if zero_vol.any():
            group = (zero_vol != zero_vol.shift()).cumsum()
            consecutive = zero_vol.groupby(group).transform("sum")
            to_interp = zero_vol & (consecutive >= 3)
            if to_interp.any():
                df = df.copy()
                df.loc[to_interp, "volume"] = np.nan
                df["volume"] = df["volume"].interpolate(method="linear")

        return df, report

    # ------------------------------------------------------------------
    # step4: 가격 이상치 — IQR + Z-score + 순간변화율 (2/3 이상 해당 시 확정)
    # ------------------------------------------------------------------

    def step4_price_outliers(
        self, df: pd.DataFrame, report: QualityReport
    ) -> tuple[pd.DataFrame, QualityReport]:
        if len(df) < 10:
            return df, report

        close = df["close"].astype(float)

        # IQR 1.5배
        q1, q3 = close.quantile(0.25), close.quantile(0.75)
        iqr = q3 - q1
        iqr_mask = (close < q1 - 1.5 * iqr) | (close > q3 + 1.5 * iqr)

        # Z-score > 3
        std = close.std()
        z_mask = ((close - close.mean()) / (std + 1e-9)).abs() > 3

        # 순간 변화율 > 15%
        pct_mask = close.pct_change().abs() > 0.15

        outlier_mask = (
            iqr_mask.astype(int) + z_mask.astype(int) + pct_mask.astype(int)
        ) >= 2
        report.price_outlier_count = int(outlier_mask.sum())

        if outlier_mask.any():
            df = df.copy()
            median_price = float(close.median())
            df.loc[outlier_mask, "close"] = median_price
            # high/low도 median 기준 보정
            df.loc[outlier_mask, "high"] = df.loc[outlier_mask, "high"].clip(upper=median_price * 1.001)
            df.loc[outlier_mask, "low"] = df.loc[outlier_mask, "low"].clip(lower=median_price * 0.999)
            report.warnings.append(f"가격 이상치 {report.price_outlier_count}개 median 대체")

        return df, report

    # ------------------------------------------------------------------
    # step5: IsolationForest 이상 감지
    # ------------------------------------------------------------------

    async def step5_anomaly_detection(
        self, df: pd.DataFrame, report: QualityReport
    ) -> tuple[pd.DataFrame, QualityReport]:
        if len(df) < 50:   # 데이터 부족 시 스킵
            return df, report

        required = ["close", "volume", "high", "low", "open"]
        if not all(c in df.columns for c in required):
            return df, report

        feats = pd.DataFrame(index=df.index)
        feats["close_pct_change"] = df["close"].pct_change().fillna(0)
        feats["volume_pct_change"] = df["volume"].pct_change().fillna(0)
        feats["high_low_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
        feats["candle_body_ratio"] = (
            (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-9)
        )
        feats = feats.replace([np.inf, -np.inf], 0).fillna(0)

        # IsolationForest CPU 작업 — 스레드 풀로 격리 (이벤트 루프 블로킹 방지)
        loop = asyncio.get_event_loop()
        preds = await loop.run_in_executor(None, _run_isolation_forest, feats.values)

        anomaly_mask = preds == -1
        df = df.copy()
        df["is_anomaly"] = anomaly_mask
        df["exclude_from_training"] = anomaly_mask

        report.anomaly_count = int(anomaly_mask.sum())
        report.anomaly_pct = report.anomaly_count / max(len(df), 1)

        if report.anomaly_pct > 0.10:
            report.warnings.append(
                f"이상치 비율 {report.anomaly_pct:.1%} > 10% — 데이터 품질 심각"
            )

        return df, report

    # ------------------------------------------------------------------
    # step6: 데이터 신선도
    # ------------------------------------------------------------------

    def step6_freshness(
        self, df: pd.DataFrame, interval: str, report: QualityReport
    ) -> QualityReport:
        if df.empty:
            report.stale_data = True
            return report

        last_ts = df.index[-1]
        if not isinstance(last_ts, pd.Timestamp):
            last_ts = pd.Timestamp(last_ts)
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")

        now = pd.Timestamp.now(tz="UTC")
        lag = now - last_ts
        report.lag_seconds = lag.total_seconds()

        max_lag = MAX_LAG.get(interval, timedelta(minutes=10))
        if lag > max_lag * 2:
            report.stale_data = True
            report.warnings.append(
                f"데이터 심각한 지연 {lag.total_seconds():.0f}초 — Level 4 연동"
            )
            if self._cb is not None:
                self._cb.trigger(4, f"데이터 심각한 지연 {lag.total_seconds():.0f}초")
            else:
                logger.warning("[품질검증] 서킷브레이커 미주입 — Level 4 발동 스킵 (데이터 지연)")
        elif lag > max_lag:
            report.stale_data = True
            report.warnings.append(f"데이터 지연 {lag.total_seconds():.0f}초")

        return report

    # ------------------------------------------------------------------
    # step7: WebSocket vs REST 교차 검증
    # ------------------------------------------------------------------

    def step7_cross_validate(
        self, df: pd.DataFrame, ws_price: float, report: QualityReport
    ) -> QualityReport:
        if df.empty or ws_price <= 0:
            return report

        rest_close = float(df["close"].iloc[-1])
        if rest_close <= 0:
            return report

        deviation = abs(ws_price - rest_close) / (rest_close + 1e-9)
        report.price_deviation_pct = deviation * 100

        if deviation > 0.03:
            report.source_mismatch = True
            report.warnings.append(
                f"WebSocket vs REST 괴리 {deviation:.2%} > 3% — Level 4 연동"
            )
            if self._cb is not None:
                self._cb.trigger(4, f"WebSocket vs REST 괴리 {deviation:.2%}")
            else:
                logger.warning("[품질검증] 서킷브레이커 미주입 — Level 4 발동 스킵 (WS/REST 괴리)")
        elif deviation > 0.01:
            report.source_warning = True
            report.warnings.append(
                f"WebSocket vs REST 괴리 {deviation:.2%} > 1% — REST 폴백"
            )

        return report

    # ------------------------------------------------------------------
    # 점수 계산 + 유틸
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(report: QualityReport) -> float:
        score = 1.0
        score -= report.ohlcv_errors * 0.02
        score -= report.anomaly_pct * 0.01 * 100   # anomaly_pct는 0~1
        if report.stale_data:
            score -= 0.3
        if report.source_mismatch:
            score -= 0.2
        score -= report.gap_count * 0.01
        return round(max(0.0, min(1.0, score)), 4)

    @staticmethod
    def _log_report(report: QualityReport) -> None:
        level = logging.WARNING if report.score < 0.9 else logging.DEBUG
        logger.log(
            level,
            "[품질검증] %s %s score=%.3f warnings=%s",
            report.coin,
            report.interval,
            report.score,
            report.warnings,
        )

    @staticmethod
    def score_to_status(score: float) -> str:
        """점수 → 상태 문자열."""
        if score >= 0.9:
            return "NORMAL"
        if score >= 0.7:
            return "WARNING"
        if score >= 0.5:
            return "HOLD_TRAINING"
        return "STOP"

    def get_training_mask(self, df: pd.DataFrame) -> pd.Series:
        """학습 포함 여부 마스크 반환.

        - exclude_from_training==True 제외
        - 이상 구간 전후 5개 추가 제외
        """
        if "exclude_from_training" not in df.columns:
            return pd.Series([True] * len(df), index=df.index)

        mask = ~df["exclude_from_training"].fillna(False)
        # 이상 구간 전후 5개 캔들 추가 제외
        anomaly_idx = df.index[df["exclude_from_training"].fillna(False)]
        for ts in anomaly_idx:
            loc = df.index.get_loc(ts)
            start = max(0, loc - 5)
            end = min(len(df), loc + 6)
            mask.iloc[start:end] = False

        return mask
