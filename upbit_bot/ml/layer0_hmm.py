"""
ml/layer0_hmm.py — Layer 0 HMM 레짐 감지기 (Phase C)

[4 레짐]
  0: 강한 상승     — ADX>30, 수익률 높음, 변동성 보통
  1: 약한 상승·횡보 — ADX 20~30, 수익률 소폭 양수
  2: 약한 하락·횡보 — ADX<20, 수익률 소폭 음수
  3: 강한 하락     — ADX>25, 수익률 음수 큰 폭, 변동성 높음

[입력 피처 — 3개]
  log_return       : 일봉 로그 수익률
  realized_vol     : 10일 실현 변동성 (log_return rolling std)
  volume_change    : 거래량 변화율 (pct_change)

[학습 데이터]
  일봉 3년 + 1시간봉 2년 (hmmlearn 권장 200개 이상 샘플)

[폴백]
  hmmlearn 미설치 또는 학습 실패 → ADX 단독 레짐 판정
  엔진에서 HMM 오류 시 자동으로 adx_fallback=True 반환

[재학습]
  매주 일요일 자정 ProcessPoolExecutor 격리 실행 (메인 루프 블로킹 방지)
  체크포인트: models/hmm_regime.pkl

[ProcessPoolExecutor 패턴]
  # engine.py에서 호출 예시:
  # result = await loop.run_in_executor(proc_pool, detector.train_sync, df_daily, df_1h)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────
N_REGIMES: int = 4
REGIME_LABELS: dict[int, str] = {
    0: "강한상승",
    1: "약한상승횡보",
    2: "약한하락횡보",
    3: "강한하락",
}
MIN_TRAIN_SAMPLES: int = 200    # 최소 학습 샘플 수
HMM_N_ITER: int = 100           # EM 반복 횟수
CORR_TYPE: str = "full"         # GaussianHMM covariance_type

# ADX 폴백 임계값 (hmmlearn 미설치 시 사용)
ADX_STRONG_TREND: float = 30.0
ADX_NORMAL_TREND: float = 20.0
FEAR_GREED_BEAR: float = 30.0   # F&G < 30 → 레짐 3


# ─────────────────────────────────────────────────
# hmmlearn 임포트 (선택적)
# ─────────────────────────────────────────────────
try:
    from hmmlearn import hmm as _hmm_module
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    logger.warning("[HMM] hmmlearn 미설치 — ADX 폴백 모드로 동작")


# ─────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────
@dataclass
class HMMConfig:
    """HMM 하이퍼파라미터."""
    n_components: int = N_REGIMES
    covariance_type: str = CORR_TYPE
    n_iter: int = HMM_N_ITER
    random_state: int = 42
    checkpoint_dir: str = "models"
    checkpoint_name: str = "hmm_regime.pkl"
    min_train_samples: int = MIN_TRAIN_SAMPLES


@dataclass
class RegimeResult:
    """HMM 레짐 감지 결과."""
    regime: int                          # 0~3
    confidence: float                    # 최대 사후 확률 (0~1)
    adx_fallback: bool                   # True = HMM 미사용, ADX 단독
    regime_label: str                    # "강한상승" 등
    regime_probs: list[float] = field(default_factory=lambda: [0.25] * 4)

    @property
    def is_bullish(self) -> bool:
        return self.regime in (0, 1)

    @property
    def is_bearish(self) -> bool:
        return self.regime in (2, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "confidence": round(self.confidence, 4),
            "adx_fallback": self.adx_fallback,
            "regime_label": self.regime_label,
        }


# ─────────────────────────────────────────────────
# HMMRegimeDetector
# ─────────────────────────────────────────────────
class HMMRegimeDetector:
    """GaussianHMM 4레짐 감지기.

    Phase A/B: predict_adx_fallback() 자동 사용 (HMM 없이)
    Phase C: train() 후 predict() 사용

    사용법:
        detector = HMMRegimeDetector()
        detector.train(df_daily, df_1h)   # Google Colab or 주 1회
        result = detector.predict(log_return, realized_vol, volume_change)
        # result.regime → 0~3
        # result.confidence → 레짐 신뢰도
    """

    def __init__(self, config: HMMConfig | None = None) -> None:
        self._cfg = config or HMMConfig()
        self._model: Any = None          # GaussianHMM 인스턴스
        self._trained: bool = False
        self._regime_map: dict[int, int] = {}  # HMM 상태 → 레짐 인덱스 매핑

    # ─────────────────────────────────────────
    # 학습
    # ─────────────────────────────────────────
    def train(
        self,
        df_daily: Any,        # pandas DataFrame, columns: close, volume
        df_1h: Any | None = None,
    ) -> bool:
        """HMM 학습. Returns True=성공, False=실패(폴백 유지).

        Args:
            df_daily: 일봉 DataFrame (index=DatetimeIndex, close/volume 컬럼 필수)
            df_1h: 1시간봉 (현재 미사용 — 일봉 3년이 주 학습 소스)

        ProcessPoolExecutor 격리 사용 권장:
            result = await loop.run_in_executor(pool, self.train_sync, df_daily, df_1h)
        """
        if not _HMM_AVAILABLE:
            logger.warning("[HMM] hmmlearn 미설치 — 학습 불가, ADX 폴백 유지")
            return False

        try:
            import pandas as pd

            df = df_daily.copy()
            if "close" not in df.columns:
                logger.error("[HMM] 'close' 컬럼 없음")
                return False

            # 피처 생성
            df["log_return"]    = np.log(df["close"] / df["close"].shift(1))
            df["realized_vol"]  = df["log_return"].rolling(10).std()
            if "volume" in df.columns:
                df["volume_change"] = df["volume"].pct_change()
            else:
                df["volume_change"] = 0.0

            df = df[["log_return", "realized_vol", "volume_change"]].dropna()

            if len(df) < self._cfg.min_train_samples:
                logger.warning(
                    "[HMM] 학습 샘플 부족 (%d < %d)",
                    len(df), self._cfg.min_train_samples,
                )
                return False

            X = df.values.astype(np.float64)

            model = _hmm_module.GaussianHMM(
                n_components=self._cfg.n_components,
                covariance_type=self._cfg.covariance_type,
                n_iter=self._cfg.n_iter,
                random_state=self._cfg.random_state,
            )
            model.fit(X)

            # 레짐 의미 매핑: 평균 log_return 기준으로 0=고수익~3=저수익 정렬
            mean_returns = [model.means_[i][0] for i in range(self._cfg.n_components)]
            sorted_states = sorted(range(self._cfg.n_components),
                                   key=lambda i: mean_returns[i], reverse=True)
            # sorted_states[0] = 가장 높은 수익률 → 레짐 0 (강한 상승)
            self._regime_map = {state: regime for regime, state in enumerate(sorted_states)}

            self._model = model
            self._trained = True
            logger.info("[HMM] 학습 완료 (샘플=%d, 레짐매핑=%s)", len(X), self._regime_map)
            return True

        except Exception as exc:
            logger.error("[HMM] 학습 실패: %s — ADX 폴백 유지", exc)
            self._trained = False
            return False

    def train_sync(self, df_daily: Any, df_1h: Any | None = None) -> bool:
        """ProcessPoolExecutor 호환 동기 래퍼."""
        return self.train(df_daily, df_1h)

    # ─────────────────────────────────────────
    # 추론
    # ─────────────────────────────────────────
    def predict(
        self,
        log_return: float,
        realized_vol: float,
        volume_change: float,
    ) -> RegimeResult:
        """현재 시점 레짐 예측.

        학습 미완료 또는 hmmlearn 미설치 시 ADX 폴백 사용.

        Args:
            log_return: 최근 일봉 로그 수익률
            realized_vol: 10일 실현 변동성
            volume_change: 거래량 변화율

        Returns:
            RegimeResult
        """
        if not self._trained or self._model is None:
            # ADX 폴백은 adx 없이 보수적으로 레짐 1 반환
            return RegimeResult(
                regime=1, confidence=0.5, adx_fallback=True,
                regime_label=REGIME_LABELS[1],
                regime_probs=[0.1, 0.5, 0.3, 0.1],
            )

        try:
            obs = np.array([[log_return, realized_vol, volume_change]])
            log_probs = self._model.predict_proba(obs)[0]  # shape (n_components,)

            # HMM 상태 → 레짐 인덱스 변환
            regime_probs = [0.0] * N_REGIMES
            for hmm_state, prob in enumerate(log_probs):
                regime_idx = self._regime_map.get(hmm_state, hmm_state)
                if regime_idx < N_REGIMES:
                    regime_probs[regime_idx] += float(prob)

            best_regime = int(np.argmax(regime_probs))
            confidence = float(regime_probs[best_regime])

            return RegimeResult(
                regime=best_regime,
                confidence=confidence,
                adx_fallback=False,
                regime_label=REGIME_LABELS.get(best_regime, "알수없음"),
                regime_probs=regime_probs,
            )

        except Exception as exc:
            logger.warning("[HMM] 추론 실패: %s — ADX 폴백", exc)
            return RegimeResult(
                regime=1, confidence=0.5, adx_fallback=True,
                regime_label=REGIME_LABELS[1],
            )

    def predict_adx_fallback(
        self,
        adx: float,
        fear_greed: float = 50.0,
    ) -> RegimeResult:
        """ADX 단독 레짐 판정 (Phase A/B 호환, hmmlearn 불필요).

        레짐 0: ADX > 30 (강한 추세, 상승 기준)
        레짐 1: ADX 20~30 (보통 추세)
        레짐 2: ADX < 20 (횡보)
        레짐 3: Fear&Greed < 30 (공포 — 하락장)
        """
        if fear_greed < FEAR_GREED_BEAR:
            regime = 3
            conf = 0.7
            probs = [0.05, 0.10, 0.15, 0.70]
        elif adx >= ADX_STRONG_TREND:
            regime = 0
            conf = min(adx / 50.0, 0.95)
            probs = [conf, 1 - conf - 0.05, 0.05, 0.0]
            probs = [max(p, 0.0) for p in probs]
        elif adx >= ADX_NORMAL_TREND:
            regime = 1
            conf = 0.6
            probs = [0.15, 0.60, 0.20, 0.05]
        else:
            regime = 2
            conf = 0.55
            probs = [0.05, 0.20, 0.55, 0.20]

        # 정규화
        s = sum(probs)
        probs = [p / s for p in probs]

        return RegimeResult(
            regime=regime,
            confidence=conf,
            adx_fallback=True,
            regime_label=REGIME_LABELS.get(regime, "알수없음"),
            regime_probs=probs,
        )

    # ─────────────────────────────────────────
    # 체크포인트
    # ─────────────────────────────────────────
    def save(self, path: str | Path | None = None) -> None:
        """모델 + 레짐 매핑 저장."""
        if not self._trained:
            logger.warning("[HMM] 학습 미완료 — 저장 취소")
            return
        p = Path(path) if path else (
            Path(self._cfg.checkpoint_dir) / self._cfg.checkpoint_name
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "regime_map": self._regime_map,
            "config": self._cfg,
        }
        with open(p, "wb") as f:
            pickle.dump(payload, f)
        logger.info("[HMM] 체크포인트 저장: %s", p)

    def load(self, path: str | Path | None = None) -> None:
        """체크포인트 로드."""
        p = Path(path) if path else (
            Path(self._cfg.checkpoint_dir) / self._cfg.checkpoint_name
        )
        if not p.exists():
            raise FileNotFoundError(f"HMM 체크포인트 없음: {p}")
        with open(p, "rb") as f:
            payload = pickle.load(f)
        self._model = payload["model"]
        self._regime_map = payload["regime_map"]
        self._trained = True
        logger.info("[HMM] 체크포인트 로드: %s", p)

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def hmm_available(self) -> bool:
        return _HMM_AVAILABLE

    # ─────────────────────────────────────────
    # 피처 추출 유틸
    # ─────────────────────────────────────────
    @staticmethod
    def extract_features(df: Any) -> np.ndarray:
        """DataFrame → HMM 입력 피처 배열.

        Args:
            df: 일봉 DataFrame (close, volume 컬럼 필수)

        Returns:
            (n_samples, 3) ndarray — [log_return, realized_vol, volume_change]
        """
        import pandas as pd

        df = df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["realized_vol"] = df["log_return"].rolling(10).std()
        if "volume" in df.columns:
            df["volume_change"] = df["volume"].pct_change()
        else:
            df["volume_change"] = 0.0

        result = df[["log_return", "realized_vol", "volume_change"]].dropna()
        return result.values.astype(np.float64)
