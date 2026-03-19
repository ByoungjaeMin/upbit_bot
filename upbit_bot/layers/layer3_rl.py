"""
layers/layer3_rl.py — Layer 3 RL 에이전트 (DQN)

[Phase A, B] 시뮬레이션 모드만 실행.
실거래 전환 기준 (4가지 동시 충족):
  ① 시뮬레이션 ≥ 1,000 에피소드
  ② 시뮬레이션 샤프비율 ≥ 1.5
  ③ 최근 4주 실거래(룰기반) 승률 ≥ 55%
  ④ 자본 ≥ $1,000 (KRW 환산)
미충족 시 ValueError 발생 (안전장치).

[DQN 구성]
- Experience Replay Buffer (10,000): 상관관계 제거, 학습 안정화
- Target Network (100스텝 동기화): 이동 타겟 문제 방지
- ε-greedy (1.0→0.1, decay=0.995): 탐험 vs 활용 균형
- device=cpu 강제 (MPS 금지)

[PPO 전환 예시]
# from stable_baselines3 import PPO
# model = PPO("MlpPolicy", env); model.learn(100_000)

[State 20개]
rsi_5m, macd_5m, adx_5m, supertrend_signal, volume_ratio_5m,
rsi_1h, trend_dir_1h, ema50_ratio_1d, fear_greed_norm,
btc_dominance_norm, sentiment_score, kimchi_premium,
obi, tick_imbalance, position, unrealized_pnl, hold_minutes_norm,
daily_trade_ratio, atr_ratio, bb_position

[Action 18개]
0:HOLD, 1:BUY_STRONG, 2:BUY_NORMAL, 3:SELL_ALL,
4:SELL_50, 5:SELL_30, 6:GRID_SETUP, 7:GRID_CANCEL,
8:DCA_INITIAL, 9~13:DCA_SAFETY_1~5, 14:DCA_EXIT,
15:PYRAMID, 16:USDT_CONVERT, 17:TRAIL_TIGHT
"""

from __future__ import annotations

import logging
import pickle
import random
from collections import deque
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 하이퍼파라미터
# ------------------------------------------------------------------

STATE_DIM: int = 20
ACTION_DIM: int = 18
REPLAY_CAPACITY: int = 10_000
TARGET_UPDATE_STEPS: int = 100    # Target Network 동기화 주기
EPSILON_START: float = 1.0
EPSILON_END: float = 0.1
EPSILON_DECAY: float = 0.995
GAMMA: float = 0.99               # 할인율
LR: float = 0.001
BATCH_SIZE: int = 32

# 수수료 + 슬리피지 (반영 비용)
TRADING_COST = 0.0025             # 0.25%

# 실거래 전환 기준
LIVE_MIN_EPISODES: int = 1_000
LIVE_MIN_SHARPE: float = 1.5
LIVE_MIN_WIN_RATE: float = 0.55
LIVE_MIN_CAPITAL_KRW: float = 1_300_000   # ~$1,000


# ------------------------------------------------------------------
# Action 열거형
# ------------------------------------------------------------------

class Action(IntEnum):
    HOLD           = 0
    BUY_STRONG     = 1    # TREND_STRONG 매수 (풀 포지션)
    BUY_NORMAL     = 2    # TREND_NORMAL 매수 (보수적)
    SELL_ALL       = 3    # 전량 매도
    SELL_50        = 4    # 50% 부분 익절
    SELL_30        = 5    # 30% 부분 익절
    GRID_SETUP     = 6    # 그리드 설정
    GRID_CANCEL    = 7    # 그리드 해제
    DCA_INITIAL    = 8    # DCA 초기 매수
    DCA_SAFETY_1   = 9
    DCA_SAFETY_2   = 10
    DCA_SAFETY_3   = 11
    DCA_SAFETY_4   = 12
    DCA_SAFETY_5   = 13
    DCA_EXIT       = 14   # DCA 전량 익절
    PYRAMID        = 15   # 피라미딩 추가 매수
    USDT_CONVERT   = 16   # 전량 USDT 전환 (긴급)
    TRAIL_TIGHT    = 17   # 트레일링 스탑 타이트 조정


# ------------------------------------------------------------------
# PyTorch DQN 네트워크 (device=cpu 강제)
# ------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch 미설치 — DQN 비활성화 (룰기반 폴백 사용)")

if _TORCH_AVAILABLE:
    class _DQNNet(nn.Module):
        """3층 MLP DQN 네트워크 (State→Q값 18개).

        왜 3층 MLP? LSTM은 순환 의존성으로 Experience Replay와
        상성이 나쁨. MLP는 샘플 독립성 유지 → 학습 안정화.
        """

        def __init__(
            self,
            state_dim: int = STATE_DIM,
            action_dim: int = ACTION_DIM,
            hidden: int = 128,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

else:
    class _DQNNet:  # type: ignore[no-redef]
        pass


# ------------------------------------------------------------------
# Experience Replay Buffer
# ------------------------------------------------------------------

class _ReplayBuffer:
    """Experience Replay Buffer (capacity=10,000).

    왜 필요? 연속 상태 사이의 상관관계 → TD 오류 발산 방지.
    랜덤 샘플링으로 i.i.d. 배치 형성.
    """

    def __init__(self, capacity: int = REPLAY_CAPACITY) -> None:
        self._buf: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ------------------------------------------------------------------
# DQN Agent
# ------------------------------------------------------------------

class DQNAgent:
    """DQN 에이전트.

    왜 Target Network? 온라인 네트워크로 TD 타겟을 동시 계산하면
    타겟이 이동해 학습 발산. 주기적 동기화(100스텝)로 안정화.

    사용법:
        agent = DQNAgent()
        state = env.reset()
        action = agent.select_action(state)
        next_s, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_s, done)
        if len(agent.buffer) >= BATCH_SIZE:
            agent.replay()
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        lr: float = LR,
        gamma: float = GAMMA,
        epsilon: float = EPSILON_START,
    ) -> None:
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._gamma = gamma
        self._epsilon = epsilon
        self._steps = 0

        self.buffer = _ReplayBuffer(REPLAY_CAPACITY)

        if _TORCH_AVAILABLE:
            # 온라인 네트워크: 매 스텝 업데이트
            self._online = _DQNNet(state_dim, action_dim)
            # 타겟 네트워크: 100스텝마다 동기화 (안정적 TD 타겟 제공)
            self._target = _DQNNet(state_dim, action_dim)
            self._target.load_state_dict(self._online.state_dict())
            self._target.eval()
            self._opt = optim.Adam(self._online.parameters(), lr=lr)
            self._loss_fn = nn.MSELoss()
        else:
            self._online = self._target = None

    # ------------------------------------------------------------------
    # 행동 선택 (ε-greedy)
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy 탐험.

        왜 ε-greedy? 탐험(random) vs 활용(argmax Q) 균형.
        초기 ε=1.0 → 완전 탐험 / 점차 0.1로 수렴 (decay=0.995).
        """
        # ε 감소
        self._epsilon = max(
            EPSILON_END,
            self._epsilon * EPSILON_DECAY,
        )

        if not _TORCH_AVAILABLE or self._online is None:
            return random.randint(0, self._action_dim - 1)

        if random.random() < self._epsilon:
            return random.randint(0, self._action_dim - 1)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q = self._online(s)
            return int(q.argmax().item())

    # ------------------------------------------------------------------
    # 경험 저장
    # ------------------------------------------------------------------

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # 학습 (Experience Replay)
    # ------------------------------------------------------------------

    def replay(self, batch_size: int = BATCH_SIZE) -> float | None:
        """미니배치 TD 업데이트.

        왜 Replay? 순서 데이터의 시계열 상관 제거 → 과거 경험 재사용.
        TD 타겟: r + γ × max Q_target(s') (done이면 r만)
        """
        if not _TORCH_AVAILABLE or self._online is None:
            return None
        if len(self.buffer) < batch_size:
            return None

        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.tensor(np.array(states), dtype=torch.float32)
        a = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        r = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        ns = torch.tensor(np.array(next_states), dtype=torch.float32)
        d = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # 현재 Q값
        q_vals = self._online(s).gather(1, a)

        # TD 타겟 (타겟 네트워크로 계산 — 이동 타겟 방지)
        with torch.no_grad():
            q_next = self._target(ns).max(1, keepdim=True)[0]
            td_target = r + self._gamma * q_next * (1 - d)

        loss = self._loss_fn(q_vals, td_target)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

        self._steps += 1
        # Target Network 동기화 (100스텝마다)
        if self._steps % TARGET_UPDATE_STEPS == 0:
            self.update_target()

        return float(loss.item())

    def update_target(self) -> None:
        """온라인 → 타겟 네트워크 가중치 동기화."""
        if _TORCH_AVAILABLE and self._online and self._target:
            self._target.load_state_dict(self._online.state_dict())

    # ------------------------------------------------------------------
    # 체크포인트
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        meta = {
            "epsilon": self._epsilon,
            "steps": self._steps,
            "state_dim": self._state_dim,
            "action_dim": self._action_dim,
        }
        with open(p / "dqn_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        if _TORCH_AVAILABLE and self._online:
            torch.save(self._online.state_dict(), p / "dqn_online.pth")
            torch.save(self._target.state_dict(), p / "dqn_target.pth")

        logger.info("[DQN] 체크포인트 저장: %s (ε=%.3f)", p, self._epsilon)

    def load_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        meta_path = p / "dqn_meta.pkl"
        if not meta_path.exists():
            logger.warning("[DQN] 체크포인트 없음: %s", meta_path)
            return

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self._epsilon = meta.get("epsilon", EPSILON_START)
        self._steps = meta.get("steps", 0)

        if _TORCH_AVAILABLE and self._online:
            online_p = p / "dqn_online.pth"
            target_p = p / "dqn_target.pth"
            if online_p.exists():
                self._online.load_state_dict(
                    torch.load(online_p, map_location="cpu", weights_only=True)
                )
            if target_p.exists():
                self._target.load_state_dict(
                    torch.load(target_p, map_location="cpu", weights_only=True)
                )

        logger.info("[DQN] 체크포인트 로딩 완료: %s (ε=%.3f)", p, self._epsilon)

    @property
    def epsilon(self) -> float:
        return self._epsilon


# ------------------------------------------------------------------
# Trading Environment (Gym 인터페이스)
# ------------------------------------------------------------------

class TradingEnvironment:
    """DQN / PPO 공용 트레이딩 시뮬레이션 환경.

    Gym 인터페이스 준수: reset() / step(action) → (next_state, reward, done, info)
    PPO 전환 예시:
        # from stable_baselines3 import PPO
        # model = PPO("MlpPolicy", env); model.learn(100_000)

    State 20개:
        rsi_5m, macd_5m, adx_5m, supertrend_signal, volume_ratio_5m,
        rsi_1h, trend_dir_1h, ema50_ratio_1d, fear_greed_norm,
        btc_dominance_norm, sentiment_score, kimchi_premium,
        obi, tick_imbalance, position, unrealized_pnl, hold_minutes_norm,
        daily_trade_ratio, atr_ratio, bb_position
    """

    def __init__(
        self,
        price_data: list[dict[str, Any]],   # OHLCV + 피처 dict 리스트
        initial_capital: float = 1_000_000,
        daily_trade_limit: int = 10,
    ) -> None:
        if not price_data:
            raise ValueError("price_data가 비어있음")

        self._data = price_data
        self._initial_capital = initial_capital
        self._daily_limit = daily_trade_limit

        # 에피소드 상태
        self._idx: int = 0
        self._capital: float = initial_capital
        self._position: float = 0.0          # 보유 코인 수량
        self._entry_price: float = 0.0
        self._hold_steps: int = 0
        self._trades_today: int = 0
        self._episode_returns: list[float] = []
        self._total_reward: float = 0.0

    def reset(self) -> np.ndarray:
        """에피소드 초기화 → 초기 state 반환."""
        self._idx = 0
        self._capital = self._initial_capital
        self._position = 0.0
        self._entry_price = 0.0
        self._hold_steps = 0
        self._trades_today = 0
        self._episode_returns = []
        self._total_reward = 0.0
        return self._get_state()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """한 스텝 실행.

        Returns:
            (next_state, reward, done, info)
        """
        row = self._data[self._idx]
        price = float(row.get("close", row.get("close_5m", 0.0)))
        reward = self._execute_action(action, price)

        self._hold_steps += 1 if self._position > 0 else 0
        self._total_reward += reward
        self._idx += 1
        done = self._idx >= len(self._data) - 1

        next_state = self._get_state() if not done else np.zeros(STATE_DIM)

        info = {
            "capital": self._capital,
            "position": self._position,
            "total_reward": self._total_reward,
            "trades_today": self._trades_today,
            "step": self._idx,
        }
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    # 보상 설계
    # ------------------------------------------------------------------

    def _execute_action(self, action: int, price: float) -> float:
        """행동 실행 + 보상 계산.

        보상 설계:
          익절:              +수익률
          손절:              -손실률 × 1.5 (패널티 강화)
          USDT 전환 후 하락회피: +0.05
          관망 후 기회손실(>10%): -0.05
          그리드 체결:       +체결수익
          DCA 익절:          +수익률
          수수료:            -0.25% (매 거래)
        """
        if price <= 0:
            return 0.0

        reward = 0.0
        a = Action(action) if action < len(Action) else Action.HOLD

        if a in (Action.BUY_STRONG, Action.BUY_NORMAL, Action.DCA_INITIAL,
                 Action.DCA_SAFETY_1, Action.DCA_SAFETY_2, Action.DCA_SAFETY_3,
                 Action.DCA_SAFETY_4, Action.DCA_SAFETY_5, Action.PYRAMID):
            reward = self._do_buy(price, a)

        elif a in (Action.SELL_ALL, Action.DCA_EXIT, Action.USDT_CONVERT):
            reward = self._do_sell_all(price)

        elif a == Action.SELL_50:
            reward = self._do_sell_partial(price, ratio=0.5)

        elif a == Action.SELL_30:
            reward = self._do_sell_partial(price, ratio=0.3)

        elif a == Action.HOLD:
            # 관망: 포지션 없고 가격 10%+ 상승 → 기회손실 패널티
            if self._position == 0 and self._idx + 1 < len(self._data):
                next_row = self._data[min(self._idx + 1, len(self._data) - 1)]
                next_p = float(next_row.get("close", next_row.get("close_5m", price)))
                if price > 0 and (next_p - price) / price > 0.10:
                    reward = -0.05

        return reward

    def _do_buy(self, price: float, action: Action) -> float:
        if self._trades_today >= self._daily_limit:
            return -0.01  # 일 거래 한도 초과 패널티
        alloc = 0.60 if action == Action.BUY_STRONG else 0.30
        invest = self._capital * alloc
        if invest < 1000:
            return 0.0

        qty = invest / price * (1 - TRADING_COST)
        self._position += qty
        self._capital -= invest
        if self._entry_price == 0:
            self._entry_price = price
        else:
            # 평균 진입가 계산
            total_val = self._entry_price * (self._position - qty) + price * qty
            self._entry_price = total_val / self._position if self._position > 0 else price

        self._trades_today += 1
        return -TRADING_COST  # 수수료 즉시 차감

    def _do_sell_all(self, price: float) -> float:
        if self._position <= 0:
            return 0.0
        proceeds = self._position * price * (1 - TRADING_COST)
        invested = self._position * self._entry_price
        pnl_pct = (proceeds - invested) / invested if invested > 0 else 0.0
        reward = pnl_pct if pnl_pct >= 0 else pnl_pct * 1.5
        self._capital += proceeds
        self._position = 0.0
        self._entry_price = 0.0
        self._hold_steps = 0
        self._trades_today += 1
        self._episode_returns.append(pnl_pct)
        return reward

    def _do_sell_partial(self, price: float, ratio: float) -> float:
        if self._position <= 0:
            return 0.0
        sell_qty = self._position * ratio
        proceeds = sell_qty * price * (1 - TRADING_COST)
        invested = sell_qty * self._entry_price
        pnl_pct = (proceeds - invested) / invested if invested > 0 else 0.0
        reward = pnl_pct if pnl_pct >= 0 else pnl_pct * 1.5
        self._capital += proceeds
        self._position -= sell_qty
        self._trades_today += 1
        return reward

    # ------------------------------------------------------------------
    # 상태 벡터 생성
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """현재 idx의 데이터 + 포지션 정보 → 20차원 상태 벡터."""
        if self._idx >= len(self._data):
            return np.zeros(STATE_DIM, dtype=np.float32)

        row = self._data[self._idx]

        def _f(key: str, default: float = 0.0) -> float:
            val = row.get(key, default)
            return float(val) if val is not None else default

        price = _f("close", _f("close_5m", 1.0))
        ema50 = _f("ema50_1d")
        ema200 = _f("ema200_1d")
        bb_upper = _f("bb_upper_5m")
        bb_lower = _f("bb_lower_5m")

        state = np.array([
            _f("rsi_5m", 50.0) / 100.0,
            np.clip(_f("macd_5m"), -1, 1),
            _f("adx_5m") / 100.0,
            float(_f("supertrend_signal")),
            np.clip(_f("volume_ratio_5m", 1.0), 0, 5) / 5.0,
            _f("rsi_1h", 50.0) / 100.0,
            float(_f("trend_dir_1h")),
            (ema50 / ema200 - 1.0) if ema200 > 0 else 0.0,
            _f("fear_greed", 50.0) / 100.0,
            _f("btc_dominance", 50.0) / 100.0,
            np.clip(_f("sentiment_score"), -1, 1),
            np.clip(_f("kimchi_premium"), -5, 10) / 10.0,
            np.clip(_f("obi"), -1, 1),
            np.clip(_f("tick_imbalance"), -1, 1),
            1.0 if self._position > 0 else 0.0,   # 포지션 보유 여부
            # 미실현 손익 (클리핑 -50%~+100%)
            np.clip(
                (price / self._entry_price - 1.0) if self._entry_price > 0 else 0.0,
                -0.5, 1.0,
            ),
            min(self._hold_steps / 60.0, 1.0),    # 보유 시간 (60스텝 정규화)
            min(self._trades_today / self._daily_limit, 1.0),
            # ATR 비율
            _f("atr_5m") / price if price > 0 else 0.0,
            # BB 위치 ((close - lower) / (upper - lower))
            (price - bb_lower) / (bb_upper - bb_lower)
            if (bb_upper - bb_lower) > 0 else 0.5,
        ], dtype=np.float32)

        return np.clip(state, -10.0, 10.0)

    # ------------------------------------------------------------------
    # 에피소드 결과
    # ------------------------------------------------------------------

    def episode_sharpe(self) -> float:
        """에피소드 샤프비율 (연환산 기준)."""
        rets = np.array(self._episode_returns)
        if len(rets) < 2:
            return 0.0
        std = float(rets.std())
        if std < 1e-9:
            return 0.0
        return float(rets.mean()) / std * np.sqrt(252 * 24 * 12)

    def episode_win_rate(self) -> float:
        rets = np.array(self._episode_returns)
        if len(rets) == 0:
            return 0.0
        return float((rets > 0).mean())


# ------------------------------------------------------------------
# 실거래 전환 기준 체크
# ------------------------------------------------------------------

def check_live_trade_eligibility(
    sim_episodes: int,
    sim_sharpe: float,
    rule_based_win_rate: float,
    capital_krw: float,
) -> None:
    """4가지 동시 충족 여부 확인. 미충족 시 ValueError 발생 (안전장치).

    Phase A, B에서는 시뮬레이션 모드만 실행해야 함.
    이 함수를 호출하면 실거래 전환 의도이므로 엄격히 검사.
    """
    errors = []

    if sim_episodes < LIVE_MIN_EPISODES:
        errors.append(
            f"시뮬레이션 에피소드 부족: {sim_episodes} < {LIVE_MIN_EPISODES}"
        )
    if sim_sharpe < LIVE_MIN_SHARPE:
        errors.append(
            f"시뮬레이션 샤프비율 부족: {sim_sharpe:.3f} < {LIVE_MIN_SHARPE}"
        )
    if rule_based_win_rate < LIVE_MIN_WIN_RATE:
        errors.append(
            f"실거래 승률 부족: {rule_based_win_rate:.1%} < {LIVE_MIN_WIN_RATE:.1%}"
        )
    if capital_krw < LIVE_MIN_CAPITAL_KRW:
        errors.append(
            f"자본 부족: {capital_krw:,.0f}원 < {LIVE_MIN_CAPITAL_KRW:,.0f}원"
        )

    if errors:
        raise ValueError(
            "실거래 전환 기준 미충족:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    logger.info("[Layer3] 실거래 전환 기준 충족 ✓")


# ------------------------------------------------------------------
# 에피소드 시뮬레이션 헬퍼
# ------------------------------------------------------------------

def simulate_episode(
    env: TradingEnvironment,
    agent: DQNAgent,
    train: bool = True,
) -> dict[str, Any]:
    """단일 에피소드 실행 + 결과 반환.

    Args:
        env: TradingEnvironment
        agent: DQNAgent
        train: True이면 replay() 호출 (학습 모드)

    Returns:
        {total_reward, sharpe, win_rate, final_capital, trades}
    """
    state = env.reset()
    done = False
    total_loss = 0.0
    loss_count = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)

        if train:
            loss = agent.replay()
            if loss is not None:
                total_loss += loss
                loss_count += 1

        state = next_state

    return {
        "total_reward": env._total_reward,
        "sharpe": env.episode_sharpe(),
        "win_rate": env.episode_win_rate(),
        "final_capital": env._capital,
        "trades": env._trades_today,
        "avg_loss": total_loss / loss_count if loss_count > 0 else 0.0,
    }
