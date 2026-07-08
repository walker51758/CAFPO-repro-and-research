from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from cafpo_ff6_experiments import FF6_GROUP_ORDER, FF6SplitData, normalize_long_only

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym
        from gym import spaces
    except ImportError:  # pragma: no cover
        gym = None
        spaces = None


@dataclass(frozen=True)
class FF6GroupReturns:
    returns: np.ndarray
    mask: np.ndarray
    total_cap: np.ndarray
    valid_stocks: np.ndarray
    labels: list[str]


def _group_labels() -> list[str]:
    return [f"{size}_{value}" for size, value in FF6_GROUP_ORDER]


def _slot_group_ids(data: FF6SplitData) -> np.ndarray:
    labels = _group_labels()
    group_to_id = {label: i for i, label in enumerate(labels)}
    stock_groups = data.stock_groups.copy()
    stock_groups["order_book_id"] = stock_groups["order_book_id"].astype(str)
    stock_to_group = stock_groups.set_index("order_book_id")["ff6_group"].to_dict()
    missing = [sid for sid in data.stock_order.astype(str) if sid not in stock_to_group]
    if missing:
        raise ValueError(f"{len(missing)} selected stocks are missing FF6 group labels.")
    return np.asarray([group_to_id[stock_to_group[sid]] for sid in data.stock_order.astype(str)], dtype=int)


def build_ff6_group_returns(data: FF6SplitData) -> FF6GroupReturns:
    """Build six FF6 group returns using value weights within each group."""
    labels = _group_labels()
    group_ids = _slot_group_ids(data)
    n_months = len(data.arrays.dates)
    n_groups = len(labels)
    group_returns = np.zeros((n_months, n_groups), dtype=np.float32)
    group_mask = np.zeros((n_months, n_groups), dtype=np.float32)
    group_total_cap = np.zeros((n_months, n_groups), dtype=np.float32)
    group_valid_stocks = np.zeros((n_months, n_groups), dtype=np.int32)

    for t in range(n_months):
        month_valid = data.arrays.mask[t].astype(bool)
        for group_id in range(n_groups):
            valid = month_valid & (group_ids == group_id)
            if not valid.any():
                continue
            cap = np.maximum(data.arrays.market_cap[t, valid], 0.0).astype(np.float64)
            cap_sum = float(cap.sum())
            if cap_sum <= 0:
                cap = np.ones_like(cap, dtype=np.float64)
                cap_sum = float(cap.sum())
            weights = cap / cap_sum
            group_returns[t, group_id] = float(np.dot(weights, data.arrays.ret_forward[t, valid]))
            group_mask[t, group_id] = 1.0
            group_total_cap[t, group_id] = cap_sum
            group_valid_stocks[t, group_id] = int(valid.sum())

    return FF6GroupReturns(
        returns=group_returns,
        mask=group_mask,
        total_cap=group_total_cap,
        valid_stocks=group_valid_stocks,
        labels=labels,
    )


class FF6GroupPortfolioEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        factors: np.ndarray,
        group_returns: FF6GroupReturns,
        dates: pd.DatetimeIndex,
        indices: np.ndarray,
        lookback: int = 12,
        reward_mode: Literal["log_return", "differential_sharpe", "differential_ddr"] = "log_return",
        long_only_temperature: float = 0.2,
        eta: float = 0.01,
    ) -> None:
        if spaces is None:
            raise ImportError("Install gymnasium before constructing FF6GroupPortfolioEnv.")
        self.factors = factors.astype(np.float32)
        self.group_returns = group_returns
        self.dates = dates
        self.indices = np.asarray(indices, dtype=int)
        self.lookback = lookback
        self.reward_mode = reward_mode
        self.long_only_temperature = long_only_temperature
        self.eta = eta
        self.n_factors = factors.shape[1]
        self.n_actions = len(group_returns.labels)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback, self.n_factors),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_actions,),
            dtype=np.float32,
        )
        self.ptr = 0
        self.a_ema = 0.0
        self.b_ema = 0.0
        self.dd2_ema = 0.0

    def _obs_for_month(self, month_idx: int) -> np.ndarray:
        start = month_idx - self.lookback + 1
        obs = np.zeros((self.lookback, self.n_factors), dtype=np.float32)
        src_start = max(0, start)
        src = self.factors[src_start : month_idx + 1]
        obs[-len(src) :] = src
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            np.random.seed(seed)
        self.ptr = 0
        self.a_ema = 0.0
        self.b_ema = 0.0
        self.dd2_ema = 0.0
        obs = self._obs_for_month(int(self.indices[self.ptr]))
        return obs, {}

    def step(self, action: np.ndarray):
        month_idx = int(self.indices[self.ptr])
        group_mask = self.group_returns.mask[month_idx]
        group_weights = normalize_long_only(
            action,
            group_mask,
            temperature=self.long_only_temperature,
        )
        portfolio_return = float(np.dot(group_weights, self.group_returns.returns[month_idx]))
        reward = self._reward(portfolio_return)
        self.ptr += 1
        terminated = self.ptr >= len(self.indices)
        truncated = False
        obs = (
            np.zeros((self.lookback, self.n_factors), dtype=np.float32)
            if terminated
            else self._obs_for_month(int(self.indices[self.ptr]))
        )
        info = {
            "date": self.dates[month_idx],
            "portfolio_return": portfolio_return,
            "weights": group_weights,
            "action_weights": group_weights,
            "gross_long": float(group_weights[group_weights > 0].sum()),
            "gross_short": float(group_weights[group_weights < 0].sum()),
        }
        return obs, float(reward), terminated, truncated, info

    def _reward(self, portfolio_return: float) -> float:
        if self.reward_mode == "log_return":
            return math.log1p(max(portfolio_return, -0.999999))
        if self.reward_mode == "differential_sharpe":
            delta_a = portfolio_return - self.a_ema
            delta_b = portfolio_return**2 - self.b_ema
            denom = max((self.b_ema - self.a_ema**2) ** 1.5, 1e-8)
            reward = (self.b_ema * delta_a - 0.5 * self.a_ema * delta_b) / denom
            self.a_ema += self.eta * delta_a
            self.b_ema += self.eta * delta_b
            return reward
        if self.reward_mode == "differential_ddr":
            downside = min(portfolio_return, 0.0) ** 2
            denom = max(self.dd2_ema ** 1.5, 1e-8)
            if portfolio_return > 0:
                reward = portfolio_return - 0.5 * self.a_ema
            else:
                reward = (
                    self.dd2_ema * (portfolio_return - 0.5 * self.a_ema)
                    - 0.5 * self.a_ema * portfolio_return**2
                ) / denom
            self.a_ema += self.eta * (portfolio_return - self.a_ema)
            self.dd2_ema += self.eta * (downside - self.dd2_ema)
            return reward
        raise ValueError(f"unknown reward_mode: {self.reward_mode}")


def run_ff6_group_weight_baseline(
    name: str,
    group_returns: FF6GroupReturns,
    dates: pd.DatetimeIndex,
    test_idx: np.ndarray,
    value_weighted: bool = False,
) -> pd.DataFrame:
    rows = []
    for idx in test_idx:
        valid = group_returns.mask[idx].astype(bool)
        weights = np.zeros(len(group_returns.labels), dtype=np.float32)
        if valid.any():
            if value_weighted:
                cap = np.maximum(group_returns.total_cap[idx, valid], 0.0)
                if cap.sum() <= 0:
                    cap = np.ones_like(cap, dtype=np.float32)
                weights[valid] = cap / cap.sum()
            else:
                weights[valid] = 1.0 / valid.sum()
        rows.append(
            {
                "method": name,
                "date": dates[idx],
                "portfolio_return": float(np.dot(weights, group_returns.returns[idx])),
                "valid_groups": int(valid.sum()),
                "valid_stocks": int(group_returns.valid_stocks[idx, valid].sum()) if valid.any() else 0,
            }
        )
    return pd.DataFrame(rows)


def ff6_group_diagnostics(group_returns: FF6GroupReturns, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for t, date in enumerate(dates):
        for group_id, label in enumerate(group_returns.labels):
            rows.append(
                {
                    "date": date,
                    "ff6_group": label,
                    "group_return": float(group_returns.returns[t, group_id]),
                    "valid": bool(group_returns.mask[t, group_id]),
                    "valid_stocks": int(group_returns.valid_stocks[t, group_id]),
                    "total_cap": float(group_returns.total_cap[t, group_id]),
                }
            )
    return pd.DataFrame(rows)
