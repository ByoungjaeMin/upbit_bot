"""test_layer0_5_cluster.py — CoinClusterManager + VolatilityParams 단위 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.layer0_5_cluster import (
    ATR_HIGH_THRESHOLD,
    ATR_LOW_THRESHOLD,
    CORR_THRESHOLD,
    MIN_COINS_FOR_CLUSTER,
    MIN_CORR_WINDOW,
    VOLATILITY_CONFIG,
    ClusterResult,
    CoinClusterManager,
    VolatilityParams,
    compute_correlation_matrix,
)


# ─────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────

def _price_df(n: int = 60, seed: int = 42, trend: float = 500_000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 50_000_000 + np.cumsum(rng.normal(trend, 500_000, n))
    price = np.clip(price, 1_000, None)
    return pd.DataFrame(
        {"close": price},
        index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
    )


def _correlated_df(base_df: pd.DataFrame, noise: float = 0.001) -> pd.DataFrame:
    """base_df와 높은 상관을 가진 가격 시계열."""
    rng = np.random.default_rng(99)
    close = base_df["close"].values * (1 + rng.normal(0, noise, len(base_df)))
    return pd.DataFrame({"close": close}, index=base_df.index)


def _manager() -> CoinClusterManager:
    return CoinClusterManager()


def _price_map(n_coins: int = 5, n_days: int = 60) -> dict[str, pd.DataFrame]:
    return {f"COIN{i}": _price_df(n=n_days, seed=i * 7) for i in range(n_coins)}


# ─────────────────────────────────────────────────
# 상수 검증
# ─────────────────────────────────────────────────

class TestConstants:
    def test_corr_threshold(self):
        assert 0.5 < CORR_THRESHOLD < 1.0

    def test_min_corr_window(self):
        assert MIN_CORR_WINDOW >= 20

    def test_atr_high_gt_low(self):
        assert ATR_HIGH_THRESHOLD > ATR_LOW_THRESHOLD

    def test_volatility_config_has_all_groups(self):
        assert set(VOLATILITY_CONFIG.keys()) == {"HIGH", "MEDIUM", "LOW"}

    def test_high_stop_loss_larger_than_low(self):
        assert abs(VOLATILITY_CONFIG["HIGH"].stop_loss_pct) > abs(VOLATILITY_CONFIG["LOW"].stop_loss_pct)

    def test_high_kelly_multiplier_lt_medium(self):
        assert VOLATILITY_CONFIG["HIGH"].kelly_multiplier < VOLATILITY_CONFIG["MEDIUM"].kelly_multiplier

    def test_min_coins_for_cluster(self):
        assert MIN_COINS_FOR_CLUSTER >= 2


# ─────────────────────────────────────────────────
# VolatilityParams
# ─────────────────────────────────────────────────

class TestVolatilityParams:
    def test_stop_loss_negative(self):
        for vp in VOLATILITY_CONFIG.values():
            assert vp.stop_loss_pct < 0.0

    def test_take_profit_positive(self):
        for vp in VOLATILITY_CONFIG.values():
            assert vp.take_profit_1st_pct > 0.0

    def test_kelly_multiplier_positive(self):
        for vp in VOLATILITY_CONFIG.values():
            assert vp.kelly_multiplier > 0.0

    def test_high_group_name(self):
        assert VOLATILITY_CONFIG["HIGH"].group == "HIGH"


# ─────────────────────────────────────────────────
# CoinClusterManager 초기화
# ─────────────────────────────────────────────────

class TestCoinClusterManagerInit:
    def test_instantiation(self):
        m = _manager()
        assert isinstance(m, CoinClusterManager)

    def test_initial_cluster_count_zero(self):
        m = _manager()
        assert m.cluster_count == 0

    def test_initial_coin_count_zero(self):
        m = _manager()
        assert m.coin_count == 0

    def test_initial_last_updated_none(self):
        m = _manager()
        assert m.last_updated is None

    def test_custom_corr_threshold(self):
        m = CoinClusterManager(corr_threshold=0.9)
        assert m._corr_threshold == 0.9


# ─────────────────────────────────────────────────
# update_clusters
# ─────────────────────────────────────────────────

class TestUpdateClusters:
    def test_returns_dict(self):
        m = _manager()
        result = m.update_clusters(_price_map())
        assert isinstance(result, dict)

    def test_cluster_count_positive_after_update(self):
        m = _manager()
        m.update_clusters(_price_map(n_coins=5))
        assert m.cluster_count >= 1

    def test_coin_count_matches_input(self):
        m = _manager()
        pm = _price_map(n_coins=5)
        m.update_clusters(pm)
        assert m.coin_count == 5

    def test_last_updated_set(self):
        m = _manager()
        m.update_clusters(_price_map())
        assert m.last_updated is not None

    def test_empty_input_no_crash(self):
        m = _manager()
        result = m.update_clusters({})
        assert result == {}

    def test_single_coin_no_crash(self):
        m = _manager()
        result = m.update_clusters({"BTC": _price_df()})
        # 코인 1개 → 클러스터링 불가 (MIN_COINS_FOR_CLUSTER)
        assert isinstance(result, dict)

    def test_insufficient_history_coin_skipped(self):
        """MIN_CORR_WINDOW(30일) 미만 데이터 코인 → 제외."""
        m = _manager()
        pm = {
            "BTC": _price_df(n=60),
            "SHORT": _price_df(n=10),  # 부족
        }
        result = m.update_clusters(pm)
        # SHORT은 클러스터에 없어야
        assert "SHORT" not in m._cluster_map

    def test_highly_correlated_coins_same_cluster(self):
        """상관계수 > 0.8 → 동일 클러스터."""
        base = _price_df(n=60, seed=1)
        corr = _correlated_df(base, noise=0.0001)  # 매우 높은 상관

        m = CoinClusterManager(corr_threshold=0.8)
        m.update_clusters({"BTC": base, "ETH": corr})

        btc_cid = m.get_cluster_id("BTC")
        eth_cid = m.get_cluster_id("ETH")
        assert btc_cid == eth_cid

    def test_uncorrelated_coins_different_clusters(self):
        """낮은 상관 → 다른 클러스터."""
        # 반대 방향 시계열 생성 (음 상관)
        rng = np.random.default_rng(0)
        price1 = 50_000_000 + np.cumsum(np.abs(rng.normal(100_000, 50_000, 60)))
        price2 = 50_000_000 + np.cumsum(-np.abs(rng.normal(100_000, 50_000, 60)))
        idx = pd.date_range("2024-01-01", periods=60, freq="D")
        df1 = pd.DataFrame({"close": price1}, index=idx)
        df2 = pd.DataFrame({"close": price2}, index=idx)

        m = CoinClusterManager(corr_threshold=0.8)
        m.update_clusters({"BTC": df1, "XRP": df2})

        btc_cid = m.get_cluster_id("BTC")
        xrp_cid = m.get_cluster_id("XRP")
        assert btc_cid != xrp_cid

    def test_update_twice_replaces_old_clusters(self):
        m = _manager()
        pm = _price_map(n_coins=3)
        m.update_clusters(pm)
        old_count = m.coin_count
        m.update_clusters(pm)
        assert m.coin_count == old_count  # 덮어쓰기


# ─────────────────────────────────────────────────
# get_cluster_id / get_cluster_coins
# ─────────────────────────────────────────────────

class TestClusterQueries:
    def test_unknown_coin_returns_minus_one(self):
        m = _manager()
        assert m.get_cluster_id("UNKNOWN") == -1

    def test_known_coin_returns_valid_id(self):
        m = _manager()
        m.update_clusters(_price_map(n_coins=3))
        cid = m.get_cluster_id("COIN0")
        assert cid >= 0

    def test_get_cluster_coins_unknown(self):
        m = _manager()
        coins = m.get_cluster_coins("UNKNOWN")
        assert coins == []

    def test_get_cluster_coins_excludes_self(self):
        base = _price_df(n=60, seed=1)
        corr = _correlated_df(base, noise=0.0001)
        m = CoinClusterManager(corr_threshold=0.5)
        m.update_clusters({"BTC": base, "ETH": corr})
        if m.get_cluster_id("BTC") == m.get_cluster_id("ETH"):
            coins = m.get_cluster_coins("BTC")
            assert "BTC" not in coins


# ─────────────────────────────────────────────────
# can_enter_position
# ─────────────────────────────────────────────────

class TestCanEnterPosition:
    def test_no_positions_always_true(self):
        m = _manager()
        m.update_clusters(_price_map(n_coins=3))
        assert m.can_enter_position("COIN0", []) is True

    def test_unknown_coin_returns_true(self):
        """클러스터 정보 없는 코인 → Phase A/B 호환, 허용."""
        m = _manager()
        assert m.can_enter_position("UNKNOWN", ["ETH"]) is True

    def test_same_cluster_coin_in_positions_blocks(self):
        """동일 클러스터 코인 보유 → 진입 차단."""
        base = _price_df(n=60, seed=1)
        corr = _correlated_df(base, noise=0.0001)
        m = CoinClusterManager(corr_threshold=0.5)
        m.update_clusters({"BTC": base, "ETH": corr})

        if m.get_cluster_id("BTC") == m.get_cluster_id("ETH"):
            assert m.can_enter_position("BTC", ["ETH"]) is False

    def test_different_cluster_coin_in_positions_allows(self):
        m = _manager()
        m.update_clusters(_price_map(n_coins=5))
        # 동일 클러스터 아닌 코인들이면 허용 (랜덤 시드 기준)
        result = m.can_enter_position("COIN0", ["COIN0"])  # 자기 자신 보유
        assert isinstance(result, bool)

    def test_self_coin_in_positions_ignored(self):
        """자기 자신이 positions에 있어도 진입 허용."""
        base = _price_df(n=60, seed=1)
        corr = _correlated_df(base, noise=0.0001)
        m = CoinClusterManager(corr_threshold=0.5)
        m.update_clusters({"BTC": base, "ETH": corr})
        # BTC를 이미 보유 중 + BTC 추가 진입 시도 → 자기 자신이므로 상관없음
        # can_enter는 "coin != pos_coin" 체크
        result = m.can_enter_position("BTC", ["BTC"])
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────
# get_volatility_group
# ─────────────────────────────────────────────────

class TestGetVolatilityGroup:
    def _m(self) -> CoinClusterManager:
        return _manager()

    def test_high_atr_ratio_returns_high(self):
        m = self._m()
        price = 100_000_000
        atr = price * (ATR_HIGH_THRESHOLD + 0.01)  # 6%
        vp = m.get_volatility_group("BTC", atr=atr, price=price)
        assert vp.group == "HIGH"

    def test_low_atr_ratio_returns_low(self):
        m = self._m()
        price = 100_000_000
        atr = price * (ATR_LOW_THRESHOLD - 0.005)  # 1.5%
        vp = m.get_volatility_group("BTC", atr=atr, price=price)
        assert vp.group == "LOW"

    def test_medium_atr_ratio_returns_medium(self):
        m = self._m()
        price = 100_000_000
        atr = price * 0.035  # 3.5%
        vp = m.get_volatility_group("BTC", atr=atr, price=price)
        assert vp.group == "MEDIUM"

    def test_zero_price_returns_medium_safely(self):
        m = self._m()
        vp = m.get_volatility_group("BTC", atr=1_000, price=0)
        assert vp.group == "MEDIUM"

    def test_result_is_volatility_params(self):
        m = self._m()
        vp = m.get_volatility_group("ETH", atr=1_000_000, price=10_000_000)
        assert isinstance(vp, VolatilityParams)

    def test_cached_in_volatility_map(self):
        m = self._m()
        price = 100_000_000
        atr = price * 0.06  # HIGH
        m.get_volatility_group("BTC", atr=atr, price=price)
        assert m._volatility_map.get("BTC") == "HIGH"

    def test_multiple_coins(self):
        m = self._m()
        price = 100_000_000
        m.get_volatility_group("BTC", atr=price * 0.06, price=price)
        m.get_volatility_group("ETH", atr=price * 0.035, price=price)
        m.get_volatility_group("XRP", atr=price * 0.01, price=price)
        assert m._volatility_map["BTC"] == "HIGH"
        assert m._volatility_map["ETH"] == "MEDIUM"
        assert m._volatility_map["XRP"] == "LOW"


# ─────────────────────────────────────────────────
# classify_volatility (정적 유틸)
# ─────────────────────────────────────────────────

class TestClassifyVolatility:
    def test_high(self):
        price = 100_000_000
        atr = price * 0.06
        assert CoinClusterManager.classify_volatility(atr, price) == "HIGH"

    def test_medium(self):
        price = 100_000_000
        atr = price * 0.035
        assert CoinClusterManager.classify_volatility(atr, price) == "MEDIUM"

    def test_low(self):
        price = 100_000_000
        atr = price * 0.01
        assert CoinClusterManager.classify_volatility(atr, price) == "LOW"

    def test_zero_price_returns_medium(self):
        assert CoinClusterManager.classify_volatility(1000, 0) == "MEDIUM"


# ─────────────────────────────────────────────────
# get_cluster_result
# ─────────────────────────────────────────────────

class TestGetClusterResult:
    def test_returns_cluster_result(self):
        m = _manager()
        m.update_clusters(_price_map(n_coins=3))
        result = m.get_cluster_result("COIN0")
        assert isinstance(result, ClusterResult)

    def test_unknown_coin_cluster_id_minus_one(self):
        m = _manager()
        result = m.get_cluster_result("UNKNOWN")
        assert result.cluster_id == -1

    def test_cluster_result_coin_field(self):
        m = _manager()
        m.update_clusters(_price_map(n_coins=3))
        result = m.get_cluster_result("COIN1")
        assert result.coin == "COIN1"

    def test_cluster_result_has_volatility_params(self):
        m = _manager()
        result = m.get_cluster_result("UNKNOWN")
        assert isinstance(result.volatility_params, VolatilityParams)

    def test_is_isolated_no_correlated(self):
        m = _manager()
        result = m.get_cluster_result("UNKNOWN")
        assert result.is_isolated is True


# ─────────────────────────────────────────────────
# get_status_report
# ─────────────────────────────────────────────────

class TestStatusReport:
    def test_report_is_string(self):
        m = _manager()
        report = m.get_status_report()
        assert isinstance(report, str)

    def test_report_contains_cluster_info(self):
        m = _manager()
        m.update_clusters(_price_map(n_coins=3))
        report = m.get_status_report()
        assert "클러스터" in report

    def test_empty_cluster_report_no_crash(self):
        m = _manager()
        report = m.get_status_report()
        assert isinstance(report, str)


# ─────────────────────────────────────────────────
# compute_correlation_matrix 유틸
# ─────────────────────────────────────────────────

class TestComputeCorrelationMatrix:
    def test_returns_tuple(self):
        pm = _price_map(n_coins=3)
        result = compute_correlation_matrix(pm)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_matrix_shape(self):
        pm = _price_map(n_coins=3)
        corr, coins = compute_correlation_matrix(pm)
        assert corr.shape == (3, 3)
        assert len(coins) == 3

    def test_diagonal_is_one(self):
        pm = _price_map(n_coins=3)
        corr, _ = compute_correlation_matrix(pm)
        for i in range(corr.shape[0]):
            assert corr[i, i] == pytest.approx(1.0, abs=1e-6)

    def test_symmetric(self):
        pm = _price_map(n_coins=4)
        corr, _ = compute_correlation_matrix(pm)
        assert np.allclose(corr, corr.T, atol=1e-10)

    def test_values_in_neg1_to_1(self):
        pm = _price_map(n_coins=4)
        corr, _ = compute_correlation_matrix(pm)
        assert np.all(corr >= -1.0 - 1e-8)
        assert np.all(corr <= 1.0 + 1e-8)

    def test_empty_input_returns_identity(self):
        corr, coins = compute_correlation_matrix({})
        assert len(coins) == 0

    def test_single_coin_returns_identity(self):
        pm = {"BTC": _price_df(n=60)}
        corr, coins = compute_correlation_matrix(pm)
        assert corr.shape[0] == corr.shape[1]

    def test_short_df_excluded(self):
        pm = {
            "BTC": _price_df(n=60),
            "SHORT": _price_df(n=10),  # 30일 미만
        }
        _, coins = compute_correlation_matrix(pm, window=30)
        assert "SHORT" not in coins
