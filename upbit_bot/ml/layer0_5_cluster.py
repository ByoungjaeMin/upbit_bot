"""
ml/layer0_5_cluster.py — Layer 0.5 코인 상관 클러스터링 + ATR 변동성 그룹화 (Phase C)

[클러스터링]
  30일 수익률 상관계수 행렬 (numpy.corrcoef)
  상관 > 0.8: 동일 클러스터 → 포지션 중복 방지 (동반 폭락 리스크)
  매주 일요일 재계산 + SQLite 저장

[ATR 변동성 그룹]
  HIGH   (ATR/Price > 5%): 손절-10%, 익절1차+15%, Kelly×0.6
  MEDIUM (ATR/Price 2~5%): 손절-7%,  익절1차+10%, Kelly×1.0  (기본)
  LOW    (ATR/Price < 2%): 손절-5%,  익절1차+8%,  Kelly×1.0

[동시 보유 제한]
  동일 클러스터에서 1개 코인만 보유 허용
  can_enter_position(): 진입 전 클러스터 충돌 검사

[사용법]
  manager = CoinClusterManager()
  manager.update_clusters(price_df_dict)   # 매주 재계산
  ok = manager.can_enter_position("XRP", current_positions=["ETH"])
  vg = manager.get_volatility_group("BTC", atr=2500000, price=85000000)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────
CORR_THRESHOLD: float = 0.8          # 동일 클러스터 기준 상관계수
MIN_CORR_WINDOW: int = 30            # 30일 상관계수 계산 기간 (일봉)
MIN_COINS_FOR_CLUSTER: int = 2       # 클러스터링 최소 코인 수

# ATR/Price 변동성 임계값
ATR_HIGH_THRESHOLD: float = 0.05    # ATR/Price > 5% → HIGH
ATR_LOW_THRESHOLD: float = 0.02     # ATR/Price < 2% → LOW


# ─────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────
@dataclass
class VolatilityParams:
    """ATR 변동성 그룹별 거래 파라미터."""
    group: str                    # "HIGH" | "MEDIUM" | "LOW"
    stop_loss_pct: float          # 손절 (음수)
    take_profit_1st_pct: float    # 1차 익절 (양수)
    kelly_multiplier: float       # Kelly 배율


# 그룹별 파라미터 사전 정의
VOLATILITY_CONFIG: dict[str, VolatilityParams] = {
    "HIGH": VolatilityParams(
        group="HIGH",
        stop_loss_pct=-0.10,
        take_profit_1st_pct=0.15,
        kelly_multiplier=0.6,
    ),
    "MEDIUM": VolatilityParams(
        group="MEDIUM",
        stop_loss_pct=-0.07,
        take_profit_1st_pct=0.10,
        kelly_multiplier=1.0,
    ),
    "LOW": VolatilityParams(
        group="LOW",
        stop_loss_pct=-0.05,
        take_profit_1st_pct=0.08,
        kelly_multiplier=1.0,
    ),
}


@dataclass
class ClusterResult:
    """단일 코인의 클러스터 + 변동성 정보."""
    coin: str
    cluster_id: int                      # 0부터 시작 (-1 = 미분류)
    correlated_coins: list[str] = field(default_factory=list)
    volatility_params: VolatilityParams = field(
        default_factory=lambda: VOLATILITY_CONFIG["MEDIUM"]
    )
    last_updated: str = ""

    @property
    def is_isolated(self) -> bool:
        """다른 코인과 상관 없는 독립 클러스터."""
        return len(self.correlated_coins) == 0


# ─────────────────────────────────────────────────
# CoinClusterManager
# ─────────────────────────────────────────────────
class CoinClusterManager:
    """코인 상관 클러스터링 + ATR 변동성 그룹 관리.

    사용법:
        manager = CoinClusterManager()
        # 매주 일요일 APScheduler 호출:
        manager.update_clusters({"BTC": df_btc, "ETH": df_eth, ...})

        # 진입 전 충돌 검사:
        if manager.can_enter_position("XRP", current_positions=["ETH"]):
            ...

        # 변동성 그룹:
        vg = manager.get_volatility_group("BTC", atr=2_500_000, price=85_000_000)
    """

    def __init__(
        self,
        corr_threshold: float = CORR_THRESHOLD,
        cache: Any = None,
    ) -> None:
        self._corr_threshold = corr_threshold
        self._cache = cache
        # coin → cluster_id
        self._cluster_map: dict[str, int] = {}
        # cluster_id → [coins]
        self._clusters: dict[int, list[str]] = {}
        # coin → ATR 변동성 그룹 문자열
        self._volatility_map: dict[str, str] = {}
        self._last_updated: datetime | None = None

    # ─────────────────────────────────────────
    # 클러스터 갱신 (매주 실행)
    # ─────────────────────────────────────────
    def update_clusters(
        self,
        price_dfs: dict[str, Any],
        window: int = MIN_CORR_WINDOW,
    ) -> dict[int, list[str]]:
        """30일 수익률 상관계수 기반 클러스터 갱신.

        Args:
            price_dfs: {coin: DataFrame(close 컬럼, DatetimeIndex)}
            window: 수익률 계산 기간 (일봉 기준, 기본 30)

        Returns:
            {cluster_id: [coin, ...]}
        """
        coins = [c for c, df in price_dfs.items() if df is not None and len(df) >= window]

        if len(coins) < MIN_COINS_FOR_CLUSTER:
            logger.warning("[Cluster] 코인 수 부족 (%d) — 클러스터링 건너뜀", len(coins))
            return {}

        # 수익률 행렬 (coin × days)
        returns: dict[str, np.ndarray] = {}
        for coin in coins:
            df = price_dfs[coin]
            try:
                close = df["close"].values.astype(np.float64)
                ret = np.diff(np.log(close + 1e-10))[-window:]
                if len(ret) == window:
                    returns[coin] = ret
            except Exception as exc:
                logger.warning("[Cluster] %s 수익률 계산 실패: %s", coin, exc)

        valid_coins = list(returns.keys())
        if len(valid_coins) < MIN_COINS_FOR_CLUSTER:
            return {}

        # 상관계수 행렬
        mat = np.array([returns[c] for c in valid_coins])   # (n_coins, window)
        try:
            corr_matrix = np.corrcoef(mat)                  # (n_coins, n_coins)
        except Exception as exc:
            logger.error("[Cluster] 상관계수 계산 실패: %s", exc)
            return {}

        # Union-Find로 클러스터 생성
        parent = list(range(len(valid_coins)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for i in range(len(valid_coins)):
            for j in range(i + 1, len(valid_coins)):
                if corr_matrix[i, j] > self._corr_threshold:
                    union(i, j)

        # 클러스터 맵 구성
        root_to_id: dict[int, int] = {}
        cluster_id_counter = 0
        new_cluster_map: dict[str, int] = {}
        new_clusters: dict[int, list[str]] = {}

        for idx, coin in enumerate(valid_coins):
            root = find(idx)
            if root not in root_to_id:
                root_to_id[root] = cluster_id_counter
                cluster_id_counter += 1
            cid = root_to_id[root]
            new_cluster_map[coin] = cid
            new_clusters.setdefault(cid, []).append(coin)

        self._cluster_map = new_cluster_map
        self._clusters = new_clusters
        self._last_updated = datetime.now(timezone.utc)

        logger.info(
            "[Cluster] 갱신 완료: %d개 코인 → %d개 클러스터 (임계값=%.2f)",
            len(valid_coins), len(new_clusters), self._corr_threshold,
        )

        self._save_to_db()
        return new_clusters

    # ─────────────────────────────────────────
    # 클러스터 조회
    # ─────────────────────────────────────────
    def get_cluster_id(self, coin: str) -> int:
        """코인의 클러스터 ID 반환. 미분류 시 -1."""
        return self._cluster_map.get(coin, -1)

    def get_cluster_coins(self, coin: str) -> list[str]:
        """동일 클러스터에 속한 다른 코인 목록."""
        cid = self.get_cluster_id(coin)
        if cid == -1:
            return []
        return [c for c in self._clusters.get(cid, []) if c != coin]

    def can_enter_position(
        self,
        coin: str,
        current_positions: list[str],
    ) -> bool:
        """동일 클러스터 코인 보유 여부 확인.

        Args:
            coin: 진입 후보 코인
            current_positions: 현재 보유 포지션 코인 목록

        Returns:
            True = 진입 가능 (동반 폭락 위험 없음)
            False = 동일 클러스터 코인 이미 보유 → 진입 차단
        """
        cid = self.get_cluster_id(coin)
        if cid == -1:
            # 클러스터 정보 없으면 허용 (Phase A/B 호환)
            return True

        cluster_coins = self._clusters.get(cid, [])
        for pos_coin in current_positions:
            if pos_coin != coin and pos_coin in cluster_coins:
                logger.info(
                    "[Cluster] %s 진입 차단 — 동일 클러스터(%d) 코인 %s 보유 중",
                    coin, cid, pos_coin,
                )
                return False
        return True

    def get_cluster_result(self, coin: str) -> ClusterResult:
        """ClusterResult 반환 (변동성 정보 포함)."""
        cid = self.get_cluster_id(coin)
        correlated = self.get_cluster_coins(coin)
        vol_group = self._volatility_map.get(coin, "MEDIUM")
        vp = VOLATILITY_CONFIG.get(vol_group, VOLATILITY_CONFIG["MEDIUM"])

        return ClusterResult(
            coin=coin,
            cluster_id=cid,
            correlated_coins=correlated,
            volatility_params=vp,
            last_updated=self._last_updated.isoformat() if self._last_updated else "",
        )

    # ─────────────────────────────────────────
    # ATR 변동성 그룹
    # ─────────────────────────────────────────
    def get_volatility_group(
        self,
        coin: str,
        atr: float,
        price: float,
    ) -> VolatilityParams:
        """ATR/Price 비율로 변동성 그룹 결정 + 캐시.

        Args:
            coin: 코인명
            atr: ATR(14) 값 (가격 단위)
            price: 현재가

        Returns:
            VolatilityParams (손절/익절/Kelly 파라미터 포함)
        """
        if price <= 0:
            return VOLATILITY_CONFIG["MEDIUM"]

        ratio = atr / price
        if ratio > ATR_HIGH_THRESHOLD:
            group = "HIGH"
        elif ratio < ATR_LOW_THRESHOLD:
            group = "LOW"
        else:
            group = "MEDIUM"

        self._volatility_map[coin] = group
        return VOLATILITY_CONFIG[group]

    @staticmethod
    def classify_volatility(atr: float, price: float) -> str:
        """ATR/Price → "HIGH" | "MEDIUM" | "LOW" (정적 유틸)."""
        if price <= 0:
            return "MEDIUM"
        ratio = atr / price
        if ratio > ATR_HIGH_THRESHOLD:
            return "HIGH"
        elif ratio < ATR_LOW_THRESHOLD:
            return "LOW"
        return "MEDIUM"

    # ─────────────────────────────────────────
    # 상태 조회
    # ─────────────────────────────────────────
    @property
    def cluster_count(self) -> int:
        return len(self._clusters)

    @property
    def coin_count(self) -> int:
        return len(self._cluster_map)

    @property
    def last_updated(self) -> datetime | None:
        return self._last_updated

    def get_all_clusters(self) -> dict[int, list[str]]:
        return dict(self._clusters)

    def get_status_report(self) -> str:
        """텔레그램 /cluster 명령어 응답용 리포트."""
        lines = [f"<b>📊 코인 클러스터 현황</b> (임계값={self._corr_threshold:.2f})\n"]
        for cid, coins in sorted(self._clusters.items()):
            coin_list = ", ".join(coins)
            lines.append(f"  클러스터 {cid}: [{coin_list}]")
        updated = (
            self._last_updated.strftime("%Y-%m-%d %H:%M UTC")
            if self._last_updated else "미갱신"
        )
        lines.append(f"\n최종 갱신: {updated}")
        return "\n".join(lines)

    # ─────────────────────────────────────────
    # SQLite 저장
    # ─────────────────────────────────────────
    def _save_to_db(self) -> None:
        """클러스터 결과 SQLite 저장 (coin_scan_results 등)."""
        if not self._cache:
            return
        try:
            ts = datetime.now(timezone.utc).isoformat()
            for coin, cid in self._cluster_map.items():
                vol_group = self._volatility_map.get(coin, "MEDIUM")
                row = {
                    "coin": coin,
                    "cluster_id": cid,
                    "correlated_count": len(self._clusters.get(cid, [])) - 1,
                    "volatility_group": vol_group,
                    "updated_at": ts,
                }
                self._cache.insert_or_replace("coin_cluster_log", row)
        except Exception as exc:
            logger.error("[Cluster] DB 저장 실패: %s", exc)


# ─────────────────────────────────────────────────
# 편의 함수
# ─────────────────────────────────────────────────
def compute_correlation_matrix(
    price_dfs: dict[str, Any],
    window: int = MIN_CORR_WINDOW,
) -> tuple[np.ndarray, list[str]]:
    """상관계수 행렬 단독 계산 유틸.

    Returns:
        (corr_matrix (n×n), coins_list)
    """
    returns: dict[str, np.ndarray] = {}
    for coin, df in price_dfs.items():
        if df is None or len(df) < window:
            continue
        try:
            close = df["close"].values.astype(np.float64)
            ret = np.diff(np.log(close + 1e-10))[-window:]
            if len(ret) == window:
                returns[coin] = ret
        except Exception:
            continue

    coins = list(returns.keys())
    if len(coins) < 2:
        return np.eye(max(len(coins), 1)), coins

    mat = np.array([returns[c] for c in coins])
    return np.corrcoef(mat), coins
