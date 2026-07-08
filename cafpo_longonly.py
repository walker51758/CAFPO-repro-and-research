from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd

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


def softmax_long_only_weights(
    action: np.ndarray,
    mask: np.ndarray,
    temperature: float = 0.2,
) -> np.ndarray:
    """Convert raw actor outputs to long-only fully-invested weights."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    valid = mask.astype(bool)
    weights = np.zeros_like(action, dtype=np.float32)
    if valid.sum() == 0:
        return weights

    logits = np.asarray(action, dtype=np.float64)[valid]
    logits = np.where(np.isfinite(logits), logits, 0.0)
    logits = logits / temperature
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    denom = exp_logits.sum()
    if denom <= 0 or not np.isfinite(denom):
        weights[valid] = 1.0 / valid.sum()
    else:
        weights[valid] = (exp_logits / denom).astype(np.float32)
    return weights


class SoftmaxLongOnlyPortfolioEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        factors: np.ndarray,
        forward_returns: np.ndarray,
        mask: np.ndarray,
        dates: pd.DatetimeIndex,
        indices: np.ndarray,
        lookback: int = 12,
        temperature: float = 0.2,
        reward_mode: Literal["log_return", "differential_sharpe", "differential_ddr"] = "log_return",
        eta: float = 0.01,
    ) -> None:
        if spaces is None:
            raise ImportError("Install gymnasium before constructing SoftmaxLongOnlyPortfolioEnv.")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.factors = factors.astype(np.float32)
        self.forward_returns = forward_returns.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.dates = dates
        self.indices = np.asarray(indices, dtype=int)
        self.lookback = lookback
        self.temperature = temperature
        self.reward_mode = reward_mode
        self.eta = eta
        self.max_stocks = forward_returns.shape[1]
        self.n_factors = factors.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback, self.n_factors),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.max_stocks,),
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
        weights = softmax_long_only_weights(action, self.mask[month_idx], self.temperature)
        portfolio_return = float(np.dot(weights, self.forward_returns[month_idx]))
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
            "weights": weights,
            "gross_long": float(weights[weights > 0].sum()),
            "gross_short": float(weights[weights < 0].sum()),
            "max_weight": float(weights.max()) if len(weights) else 0.0,
            "effective_n": float(1.0 / np.square(weights).sum()) if np.square(weights).sum() > 0 else 0.0,
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


def evaluate_softmax_longonly_ensemble(
    models: list[Any],
    env: SoftmaxLongOnlyPortfolioEnv,
    deterministic: bool = True,
) -> pd.DataFrame:
    """Average raw actor actions, then apply one softmax long-only mapping."""
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        actions = []
        for model in models:
            action, _ = model.predict(obs, deterministic=deterministic)
            actions.append(np.asarray(action, dtype=np.float32))
        avg_action = np.mean(np.stack(actions, axis=0), axis=0)
        obs, reward, terminated, truncated, info = env.step(avg_action)
        done = terminated or truncated
        rows.append(
            {
                "date": info["date"],
                "portfolio_return": info["portfolio_return"],
                "reward": reward,
                "gross_long": info["gross_long"],
                "gross_short": info["gross_short"],
                "ensemble_members": len(models),
                "max_weight": info["max_weight"],
                "effective_n": info["effective_n"],
            }
        )
    return pd.DataFrame(rows)
