"""test_layer3_rl.py — Layer3 RL 에이전트 단위 테스트."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from layers.layer3_rl import (
    ACTION_DIM,
    BATCH_SIZE,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    STATE_DIM,
    TARGET_UPDATE_STEPS,
    Action,
    DQNAgent,
    TradingEnvironment,
    _ReplayBuffer,
    _TORCH_AVAILABLE,
    check_live_trade_eligibility,
    simulate_episode,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _make_price_data(n: int = 200, price: float = 50_000_000.0) -> list[dict]:
    """합성 가격 데이터."""
    rng = np.random.default_rng(42)
    data = []
    p = price
    for i in range(n):
        p *= 1 + rng.normal(0, 0.005)
        data.append({
            "close_5m": p,
            "close": p,
            "open_5m": p * 0.999,
            "high_5m": p * 1.001,
            "low_5m": p * 0.999,
            "volume_5m": 10.0,
            "rsi_5m": 50.0 + rng.normal(0, 10),
            "macd_5m": rng.normal(0, 0.1),
            "adx_5m": 20.0 + abs(rng.normal(0, 5)),
            "supertrend_signal": int(rng.choice([-1, 1])),
            "volume_ratio_5m": 1.0,
            "rsi_1h": 50.0,
            "trend_dir_1h": 1,
            "ema50_1d": p * 0.99,
            "ema200_1d": p * 0.98,
            "fear_greed": 50.0,
            "btc_dominance": 45.0,
            "sentiment_score": 0.0,
            "kimchi_premium": 2.0,
            "obi": 0.1,
            "tick_imbalance": 0.1,
            "atr_5m": p * 0.01,
            "bb_upper_5m": p * 1.02,
            "bb_lower_5m": p * 0.98,
        })
    return data


# ---------------------------------------------------------------------------
# Action 열거형
# ---------------------------------------------------------------------------

class TestActionEnum:
    def test_action_count_18(self):
        assert len(Action) == ACTION_DIM

    def test_hold_is_0(self):
        assert Action.HOLD == 0

    def test_usdt_convert_is_16(self):
        assert Action.USDT_CONVERT == 16

    def test_trail_tight_is_17(self):
        assert Action.TRAIL_TIGHT == 17


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_push_and_len(self):
        buf = _ReplayBuffer(capacity=100)
        s = np.zeros(STATE_DIM)
        buf.push(s, 0, 1.0, s, False)
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = _ReplayBuffer(capacity=5)
        s = np.zeros(STATE_DIM)
        for _ in range(10):
            buf.push(s, 0, 0.0, s, False)
        assert len(buf) == 5

    def test_sample_correct_count(self):
        buf = _ReplayBuffer(capacity=100)
        s = np.zeros(STATE_DIM)
        for _ in range(50):
            buf.push(s, 0, 0.0, s, False)
        batch = buf.sample(10)
        assert len(batch) == 10

    def test_sample_raises_if_insufficient(self):
        buf = _ReplayBuffer(capacity=100)
        with pytest.raises(ValueError):
            buf.sample(5)


# ---------------------------------------------------------------------------
# DQNAgent
# ---------------------------------------------------------------------------

class TestDQNAgent:
    def test_init_epsilon_start(self):
        agent = DQNAgent()
        assert agent.epsilon == pytest.approx(EPSILON_START)

    def test_select_action_range(self):
        agent = DQNAgent()
        s = np.random.rand(STATE_DIM).astype(np.float32)
        for _ in range(20):
            a = agent.select_action(s)
            assert 0 <= a < ACTION_DIM

    def test_epsilon_decays(self):
        agent = DQNAgent()
        s = np.zeros(STATE_DIM, dtype=np.float32)
        initial_eps = agent.epsilon
        for _ in range(100):
            agent.select_action(s)
        assert agent.epsilon < initial_eps

    def test_epsilon_floor(self):
        agent = DQNAgent(epsilon=EPSILON_END)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        for _ in range(1000):
            agent.select_action(s)
        assert agent.epsilon >= EPSILON_END

    def test_remember_fills_buffer(self):
        agent = DQNAgent()
        s = np.zeros(STATE_DIM, dtype=np.float32)
        for _ in range(10):
            agent.remember(s, 0, 1.0, s, False)
        assert len(agent.buffer) == 10

    def test_replay_returns_none_when_buffer_small(self):
        agent = DQNAgent()
        result = agent.replay(batch_size=BATCH_SIZE)
        assert result is None

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch 미설치")
    def test_replay_returns_loss_when_buffer_full(self):
        agent = DQNAgent()
        s = np.random.rand(STATE_DIM).astype(np.float32)
        for _ in range(BATCH_SIZE + 10):
            agent.remember(s, 0, 1.0, s, False)
        loss = agent.replay(batch_size=BATCH_SIZE)
        assert loss is not None
        assert loss >= 0.0

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch 미설치")
    def test_update_target_does_not_crash(self):
        agent = DQNAgent()
        agent.update_target()

    def test_checkpoint_save_load(self):
        agent = DQNAgent()
        agent._epsilon = 0.55
        agent._steps = 42

        with tempfile.TemporaryDirectory() as tmp:
            agent.save_checkpoint(tmp)
            agent2 = DQNAgent()
            agent2.load_checkpoint(tmp)
            assert agent2.epsilon == pytest.approx(0.55)

    def test_load_nonexistent_is_noop(self):
        agent = DQNAgent()
        agent.load_checkpoint("/nonexistent/path")
        assert agent.epsilon == pytest.approx(EPSILON_START)


# ---------------------------------------------------------------------------
# DQNNet (PyTorch 전용)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch 미설치")
class TestDQNNet:
    def test_output_shape(self):
        import torch
        from layers.layer3_rl import _DQNNet
        net = _DQNNet(STATE_DIM, ACTION_DIM)
        x = torch.randn(1, STATE_DIM)
        out = net(x)
        assert out.shape == (1, ACTION_DIM)

    def test_batch_output_shape(self):
        import torch
        from layers.layer3_rl import _DQNNet
        net = _DQNNet(STATE_DIM, ACTION_DIM)
        x = torch.randn(BATCH_SIZE, STATE_DIM)
        out = net(x)
        assert out.shape == (BATCH_SIZE, ACTION_DIM)


# ---------------------------------------------------------------------------
# TradingEnvironment
# ---------------------------------------------------------------------------

class TestTradingEnvironment:
    def test_reset_returns_state_dim(self):
        env = TradingEnvironment(_make_price_data(50))
        s = env.reset()
        assert s.shape == (STATE_DIM,)

    def test_step_returns_tuple(self):
        env = TradingEnvironment(_make_price_data(50))
        env.reset()
        ns, reward, done, info = env.step(Action.HOLD)
        assert isinstance(ns, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_done_at_end(self):
        data = _make_price_data(5)
        env = TradingEnvironment(data)
        env.reset()
        done = False
        for _ in range(len(data)):
            _, _, done, _ = env.step(Action.HOLD)
        assert done

    def test_buy_reduces_capital(self):
        env = TradingEnvironment(_make_price_data(50))
        env.reset()
        initial_cap = env._capital
        env.step(Action.BUY_STRONG)
        assert env._capital < initial_cap

    def test_sell_all_with_no_position_no_crash(self):
        env = TradingEnvironment(_make_price_data(50))
        env.reset()
        _, reward, _, _ = env.step(Action.SELL_ALL)
        assert reward == 0.0

    def test_buy_then_sell_cycle(self):
        env = TradingEnvironment(_make_price_data(50))
        env.reset()
        env.step(Action.BUY_STRONG)
        assert env._position > 0
        env.step(Action.SELL_ALL)
        assert env._position == 0.0

    def test_state_no_nan(self):
        env = TradingEnvironment(_make_price_data(50))
        s = env.reset()
        assert not np.any(np.isnan(s))

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            TradingEnvironment([])

    def test_daily_trade_limit_penalty(self):
        env = TradingEnvironment(_make_price_data(50), daily_trade_limit=1)
        env.reset()
        env.step(Action.BUY_STRONG)
        env._trades_today = 1  # 한도 소진
        # 추가 매수 → 패널티 반환
        _, reward, _, _ = env.step(Action.BUY_STRONG)
        assert reward < 0


class TestEpisodeMetrics:
    def test_episode_sharpe_no_trades_zero(self):
        env = TradingEnvironment(_make_price_data(10))
        env.reset()
        assert env.episode_sharpe() == 0.0

    def test_episode_win_rate_no_trades_zero(self):
        env = TradingEnvironment(_make_price_data(10))
        env.reset()
        assert env.episode_win_rate() == 0.0


# ---------------------------------------------------------------------------
# 실거래 전환 기준 체크
# ---------------------------------------------------------------------------

class TestLiveTradeEligibility:
    def test_all_criteria_pass(self):
        check_live_trade_eligibility(
            sim_episodes=1000,
            sim_sharpe=1.5,
            rule_based_win_rate=0.55,
            capital_krw=1_300_000,
        )  # 예외 없어야 함

    def test_episodes_insufficient_raises(self):
        with pytest.raises(ValueError, match="에피소드"):
            check_live_trade_eligibility(999, 1.5, 0.55, 1_300_000)

    def test_sharpe_insufficient_raises(self):
        with pytest.raises(ValueError, match="샤프"):
            check_live_trade_eligibility(1000, 1.4, 0.55, 1_300_000)

    def test_win_rate_insufficient_raises(self):
        with pytest.raises(ValueError, match="승률"):
            check_live_trade_eligibility(1000, 1.5, 0.54, 1_300_000)

    def test_capital_insufficient_raises(self):
        with pytest.raises(ValueError, match="자본"):
            check_live_trade_eligibility(1000, 1.5, 0.55, 999_999)

    def test_multiple_failures_combined(self):
        with pytest.raises(ValueError) as exc:
            check_live_trade_eligibility(500, 1.0, 0.4, 500_000)
        assert "에피소드" in str(exc.value)
        assert "샤프" in str(exc.value)
        assert "승률" in str(exc.value)
        assert "자본" in str(exc.value)


# ---------------------------------------------------------------------------
# simulate_episode 헬퍼
# ---------------------------------------------------------------------------

class TestSimulateEpisode:
    def test_returns_required_keys(self):
        env = TradingEnvironment(_make_price_data(30))
        agent = DQNAgent()
        result = simulate_episode(env, agent, train=False)
        for key in ("total_reward", "sharpe", "win_rate", "final_capital", "trades"):
            assert key in result

    def test_final_capital_positive(self):
        env = TradingEnvironment(_make_price_data(30), initial_capital=1_000_000)
        agent = DQNAgent()
        result = simulate_episode(env, agent, train=False)
        # 포지션 없이 HOLD만 해도 자본은 유지
        assert result["final_capital"] >= 0
