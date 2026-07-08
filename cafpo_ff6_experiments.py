from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from cafpo_reproduction import CafpoArrays, Split, normalize_long_short

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


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "rqdata_output"
PREPANEL_PATH = OUTPUT_DIR / "cafpo_82_prepanel_allstocks.parquet"
MANIFEST_PATH = OUTPUT_DIR / "cafpo_82_feature_manifest.csv"

FF6_GROUP_ORDER = [
    ("Big", "Growth"),
    ("Big", "Neutral"),
    ("Big", "Value"),
    ("Small", "Growth"),
    ("Small", "Neutral"),
    ("Small", "Value"),
]


@dataclass(frozen=True)
class FF6SplitData:
    split: Split
    selection_date: pd.Timestamp
    arrays: CafpoArrays
    stock_order: np.ndarray
    stock_groups: pd.DataFrame
    action_stock_order: np.ndarray
    action_slot_indices: np.ndarray
    action_groups: pd.DataFrame
    kept_feature_cols: list[str]
    dropped_feature_cols: list[str]
    missingness_report: pd.DataFrame
    audit: dict[str, Any]


def load_prepanel_and_manifest(
    prepanel_path: Path = PREPANEL_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre = pd.read_parquet(prepanel_path)
    pre["date"] = pd.to_datetime(pre["date"])
    if "next_date" in pre.columns:
        pre["next_date"] = pd.to_datetime(pre["next_date"])
    manifest = pd.read_csv(manifest_path)
    return pre.sort_values(["date", "order_book_id"]).reset_index(drop=True), manifest


def rolling_test_years(pre: pd.DataFrame, train_years: int = 10) -> list[int]:
    years = sorted(pd.Index(pre["date"].dt.year.unique()).tolist())
    return [year for year in years if year - train_years >= years[0]]


def _finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.where(np.isfinite(values), values, np.nan), index=series.index)


def normalize_long_only(
    action: np.ndarray,
    mask: np.ndarray,
    temperature: float = 0.2,
) -> np.ndarray:
    """Map raw action logits to long-only fully-invested softmax weights."""
    valid = mask.astype(bool)
    weights = np.zeros_like(action, dtype=np.float32)
    if valid.sum() == 0:
        return weights
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    raw = np.asarray(action, dtype=np.float64)
    raw_valid = raw[valid]
    raw_valid = np.where(np.isfinite(raw_valid), raw_valid, 0.0)
    logits = raw_valid / float(temperature)
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    denom = exp_logits.sum()
    if not np.isfinite(denom) or denom <= 1e-12:
        exp_logits = np.ones_like(raw_valid, dtype=np.float64)
        denom = exp_logits.sum()
    weights[valid] = (exp_logits / denom).astype(np.float32)
    return weights


def _assign_ff6_groups(
    eligible: pd.DataFrame,
    size_col: str,
    value_col: str,
) -> pd.DataFrame:
    out = eligible.copy()
    size = _finite_numeric(out[size_col])
    value = _finite_numeric(out[value_col])
    size_median = float(size.median())
    value_30, value_70 = value.quantile([0.3, 0.7]).astype(float).tolist()
    if not np.isfinite(size_median) or not np.isfinite(value_30) or not np.isfinite(value_70):
        raise ValueError("FF6 breakpoints are not finite.")
    if value_30 >= value_70:
        raise ValueError(f"FF6 value breakpoints are degenerate: {value_30}, {value_70}")

    out["_size_value"] = size
    out["_bm_value"] = value
    out["size_group"] = np.where(size >= size_median, "Big", "Small")
    out["value_group"] = pd.cut(
        value,
        bins=[-np.inf, value_30, value_70, np.inf],
        labels=["Growth", "Neutral", "Value"],
    ).astype(str)
    out["ff6_group"] = out["size_group"] + "_" + out["value_group"]
    out["size_break_median"] = size_median
    out["value_break_30"] = value_30
    out["value_break_70"] = value_70
    return out


def _strict_training_coverage(
    pre: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    min_train_months_ratio: float = 1.0,
) -> pd.DataFrame:
    train = pre[pre["date"].isin(train_dates)]
    n_train = len(train_dates)
    coverage = train.groupby("order_book_id").agg(
        train_rows=("date", "nunique"),
        train_ret_1m_nonnull=("ret_1m", lambda s: int(s.notna().sum())),
        train_ret_fwd_1m_nonnull=("ret_fwd_1m", lambda s: int(s.notna().sum())),
        train_feature_months=("feature_nonnull_raw", lambda s: int((s.fillna(0) > 0).sum())),
    )
    min_months = int(round(min_train_months_ratio * n_train))
    coverage["strict_train_ok"] = (
        (coverage["train_rows"] >= min_months)
        & (coverage["train_ret_1m_nonnull"] >= min_months)
        & (coverage["train_ret_fwd_1m_nonnull"] >= min_months)
        & (coverage["train_feature_months"] >= min_months)
    )
    return coverage


def _sample_balanced_ff6(
    grouped: pd.DataFrame,
    n_total: int,
    seed: int,
) -> pd.DataFrame:
    if n_total % len(FF6_GROUP_ORDER) != 0:
        raise ValueError(f"n_total must be divisible by 6, got {n_total}.")
    quota = n_total // len(FF6_GROUP_ORDER)
    rng = np.random.default_rng(seed)
    picks = []
    for size_group, value_group in FF6_GROUP_ORDER:
        cell = grouped[
            (grouped["size_group"] == size_group)
            & (grouped["value_group"] == value_group)
        ].sort_values("order_book_id")
        if len(cell) < quota:
            label = f"{size_group}_{value_group}"
            raise ValueError(f"FF6 cell {label} has {len(cell)} eligible stocks, need {quota}.")
        take = rng.choice(cell.index.to_numpy(), size=quota, replace=False)
        picks.append(grouped.loc[take])
    selected = pd.concat(picks, ignore_index=True)
    group_rank = {
        f"{size}_{value}": rank
        for rank, (size, value) in enumerate(FF6_GROUP_ORDER)
    }
    selected["_group_rank"] = selected["ff6_group"].map(group_rank)
    return selected.sort_values(["_group_rank", "order_book_id"]).drop(columns=["_group_rank"])


def _ff6_count_dict(df: pd.DataFrame) -> dict[str, int]:
    counts = df.groupby(["size_group", "value_group"]).size()
    return {f"{size}_{value}": int(count) for (size, value), count in counts.items()}


def select_ff6_universe(
    pre: pd.DataFrame,
    test_year: int,
    n_stocks: int = 180,
    train_years: int = 10,
    seed: int = 42,
    size_col: str = "mvel1",
    value_col: str = "lagged_x_bm",
    min_train_months_ratio: float = 1.0,
) -> tuple[pd.Timestamp, pd.DatetimeIndex, pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    """Select a fixed FF Size x Value/Growth universe.

    The default value breakpoint column is ``lagged_x_bm`` so the annual B/M
    signal follows the same no-lookahead availability convention as the CAFPO
    feature panel. Pass ``value_col="bm"`` explicitly for a raw same-month B/M
    formation diagnostic.

    ``min_train_months_ratio`` (default 1.0) relaxes the strict 100% training
    coverage rule. The original pipeline requires every selected stock to have
    a non-null ``ret_1m``, ``ret_fwd_1m`` and at least one raw lagged feature
    in every single training month. For longer training windows (e.g. starting
    in 2007-01, which has no ``ret_1m``), no stock can satisfy the strict rule,
    so we allow a fractional cutoff. ``1.0`` keeps the original behaviour.
    """
    train_start = test_year - train_years
    train_end = test_year - 1
    train_dates = pd.DatetimeIndex(
        sorted(pre.loc[pre["date"].dt.year.between(train_start, train_end), "date"].unique())
    )
    test_dates = pd.DatetimeIndex(sorted(pre.loc[pre["date"].dt.year.eq(test_year), "date"].unique()))
    if len(train_dates) != train_years * 12:
        raise ValueError(f"{test_year}: expected {train_years * 12} train months, got {len(train_dates)}.")
    if len(test_dates) == 0:
        raise ValueError(f"{test_year}: no test months found.")

    selection_date = pd.Timestamp(test_dates.min())
    coverage = _strict_training_coverage(pre, train_dates, min_train_months_ratio=min_train_months_ratio)
    formation = pre.loc[pre["date"].eq(selection_date)].copy()
    if size_col not in formation.columns or value_col not in formation.columns:
        raise KeyError(f"Missing FF6 formation columns: {size_col!r}, {value_col!r}.")

    formation["_size_value"] = _finite_numeric(formation[size_col])
    formation["_bm_value"] = _finite_numeric(formation[value_col])
    formation = formation[
        formation["_size_value"].notna()
        & formation["_bm_value"].notna()
        & formation["ret_fwd_1m"].notna()
    ].copy()
    formation = formation.join(coverage, on="order_book_id")
    eligible = formation[formation["strict_train_ok"].fillna(False)].copy()
    if eligible.empty:
        raise ValueError(
            f"{test_year}: no stocks satisfy strict 120-month ret_1m/ret_fwd_1m/feature coverage."
        )

    grouped = _assign_ff6_groups(eligible, "_size_value", "_bm_value")
    selected = _sample_balanced_ff6(grouped, n_total=n_stocks, seed=seed + test_year)
    return selection_date, train_dates, test_dates, grouped, selected


def _feature_missingness_report(
    pre: pd.DataFrame,
    manifest: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    stock_order: np.ndarray,
    missing_rate_threshold: float = 0.5,
) -> pd.DataFrame:
    raw_cols = manifest["lagged_raw_col"].tolist()
    safe_cols = manifest["feature_safe"].tolist()
    train = pre[
        pre["date"].isin(train_dates)
        & pre["order_book_id"].isin(stock_order)
    ][["date", "order_book_id", *raw_cols]].copy()
    records = []
    for raw_col, safe_col in zip(raw_cols, safe_cols, strict=True):
        values = _finite_numeric(train[raw_col])
        miss = values.isna()
        monthly_rate = miss.groupby(train["date"]).mean().reindex(train_dates, fill_value=1.0)
        bad_months = int((monthly_rate >= missing_rate_threshold).sum())
        records.append(
            {
                "feature_safe": safe_col,
                "lagged_raw_col": raw_col,
                "train_months": len(train_dates),
                "bad_months_ge_50pct_missing": bad_months,
                "bad_month_share": bad_months / len(train_dates),
                "mean_train_missing_rate": float(monthly_rate.mean()),
                "max_train_missing_rate": float(monthly_rate.max()),
                "drop_feature": bad_months > len(train_dates) / 2,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["drop_feature", "bad_month_share", "mean_train_missing_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _median_impute_and_rank(
    panel: pd.DataFrame,
    manifest: pd.DataFrame,
    kept_safe_cols: list[str],
) -> pd.DataFrame:
    out = panel.copy()
    kept = manifest[manifest["feature_safe"].isin(kept_safe_cols)].copy()
    for raw_col in kept["lagged_raw_col"]:
        out[raw_col] = _finite_numeric(out[raw_col])

    for raw_col in kept["lagged_raw_col"]:
        med = out.groupby("date")[raw_col].transform("median")
        out[raw_col] = out[raw_col].fillna(med)

    ranked_cols: dict[str, pd.Series] = {}
    for _, row in kept.iterrows():
        raw_col = row["lagged_raw_col"]
        safe_col = row["feature_safe"]

        def rank_to_unit(series: pd.Series) -> pd.Series:
            valid = series.notna()
            result = pd.Series(np.nan, index=series.index, dtype=float)
            if valid.sum() == 1:
                result.loc[valid] = 0.0
            elif valid.sum() > 1:
                ranks = series.loc[valid].rank(method="average")
                result.loc[valid] = 2.0 * ((ranks - 1.0) / (valid.sum() - 1.0)) - 1.0
            return result

        ranked_cols[safe_col] = out.groupby("date")[raw_col].transform(rank_to_unit).fillna(0.0)
    if ranked_cols:
        out = pd.concat([out, pd.DataFrame(ranked_cols, index=out.index)], axis=1)
    return out


def fixed_panel_to_arrays(
    panel: pd.DataFrame,
    feature_cols: list[str],
    dates: pd.DatetimeIndex,
    stock_order: np.ndarray,
) -> CafpoArrays:
    n_months = len(dates)
    n_stocks = len(stock_order)
    n_features = len(feature_cols)
    x = np.zeros((n_months, n_stocks, n_features), dtype=np.float32)
    ret_current = np.zeros((n_months, n_stocks), dtype=np.float32)
    ret_forward = np.zeros((n_months, n_stocks), dtype=np.float32)
    market_cap = np.zeros((n_months, n_stocks), dtype=np.float32)
    mask = np.zeros((n_months, n_stocks), dtype=np.float32)
    stock_ids = np.tile(stock_order.astype("<U32"), (n_months, 1))

    date_pos = {pd.Timestamp(date): i for i, date in enumerate(dates)}
    stock_pos = {str(sid): j for j, sid in enumerate(stock_order)}
    cols = ["date", "order_book_id", "ret_1m", "ret_fwd_1m", "mvel1", *feature_cols]
    for row in panel[cols].itertuples(index=False):
        date = pd.Timestamp(row.date)
        sid = str(row.order_book_id)
        if date not in date_pos or sid not in stock_pos:
            continue
        i = date_pos[date]
        j = stock_pos[sid]
        feature_values = np.asarray(row[-n_features:], dtype=np.float32)
        x[i, j] = np.where(np.isfinite(feature_values), feature_values, 0.0)
        ret_1m = float(row.ret_1m) if pd.notna(row.ret_1m) else 0.0
        ret_fwd = float(row.ret_fwd_1m) if pd.notna(row.ret_fwd_1m) else 0.0
        cap = float(row.mvel1) if pd.notna(row.mvel1) else 0.0
        ret_current[i, j] = ret_1m if np.isfinite(ret_1m) else 0.0
        ret_forward[i, j] = ret_fwd if np.isfinite(ret_fwd) else 0.0
        market_cap[i, j] = cap if np.isfinite(cap) else 0.0
        if pd.notna(row.ret_fwd_1m) and np.isfinite(ret_fwd):
            mask[i, j] = 1.0

    assert np.isfinite(x).all()
    assert np.isfinite(ret_current).all()
    assert np.isfinite(ret_forward).all()
    return CafpoArrays(
        dates=dates,
        feature_cols=feature_cols,
        x=x,
        ret_current=ret_current,
        ret_forward=ret_forward,
        market_cap=market_cap,
        mask=mask,
        stock_ids=stock_ids,
    )


def subset_arrays(arrays: CafpoArrays, slot_indices: np.ndarray) -> CafpoArrays:
    slot_indices = np.asarray(slot_indices, dtype=int)
    return CafpoArrays(
        dates=arrays.dates,
        feature_cols=arrays.feature_cols,
        x=arrays.x[:, slot_indices, :],
        ret_current=arrays.ret_current[:, slot_indices],
        ret_forward=arrays.ret_forward[:, slot_indices],
        market_cap=arrays.market_cap[:, slot_indices],
        mask=arrays.mask[:, slot_indices],
        stock_ids=arrays.stock_ids[:, slot_indices],
    )


def run_universe_weight_baseline(
    name: str,
    arrays: CafpoArrays,
    test_idx: np.ndarray,
    value_weighted: bool = False,
) -> pd.DataFrame:
    """Long-only equal/value-weighted return over the fixed valid universe each month."""
    rows = []
    for idx in test_idx:
        valid = arrays.mask[idx].astype(bool)
        weights = np.zeros(arrays.mask.shape[1], dtype=np.float32)
        if valid.any():
            if value_weighted:
                cap = np.maximum(arrays.market_cap[idx, valid], 0.0)
                if cap.sum() <= 0:
                    cap = np.ones_like(cap, dtype=np.float32)
                weights[valid] = cap / cap.sum()
            else:
                weights[valid] = 1.0 / valid.sum()
        rows.append(
            {
                "method": name,
                "date": arrays.dates[idx],
                "portfolio_return": float(np.dot(weights, arrays.ret_forward[idx])),
                "valid_stocks": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_ff6_split_data(
    pre: pd.DataFrame,
    manifest: pd.DataFrame,
    test_year: int,
    n_stocks: int = 180,
    n_action_stocks: int | None = None,
    drop_sparse_features: bool = False,
    train_years: int = 10,
    seed: int = 42,
    size_col: str = "mvel1",
    value_col: str = "lagged_x_bm",
    min_train_months_ratio: float = 1.0,
) -> FF6SplitData:
    """Build one rolling split from the all-stock prepanel only.

    ``min_train_months_ratio`` is forwarded to :func:`select_ff6_universe`.
    The default of ``1.0`` preserves the original strict 100% training
    coverage rule. Relax it (e.g. ``0.9``) when extending the training
    window into early 2007, which has no ``ret_1m`` in this dataset.
    """
    if n_action_stocks is None:
        n_action_stocks = n_stocks
    selection_date, train_dates, test_dates, eligible, selected = select_ff6_universe(
        pre=pre,
        test_year=test_year,
        n_stocks=n_stocks,
        train_years=train_years,
        seed=seed,
        size_col=size_col,
        value_col=value_col,
        min_train_months_ratio=min_train_months_ratio,
    )
    stock_order = selected["order_book_id"].astype(str).to_numpy()

    missingness = _feature_missingness_report(pre, manifest, train_dates, stock_order)
    if drop_sparse_features:
        kept_feature_cols = missingness.loc[~missingness["drop_feature"], "feature_safe"].tolist()
        dropped_feature_cols = missingness.loc[missingness["drop_feature"], "feature_safe"].tolist()
    else:
        kept_feature_cols = manifest["feature_safe"].tolist()
        dropped_feature_cols = []
    if not kept_feature_cols:
        raise ValueError(f"{test_year}: all features were dropped by missingness filter.")

    all_dates = pd.DatetimeIndex([*train_dates, *test_dates])
    needed_raw_cols = manifest.loc[
        manifest["feature_safe"].isin(kept_feature_cols), "lagged_raw_col"
    ].tolist()
    model_rows = pre[
        pre["date"].isin(all_dates)
        & pre["order_book_id"].isin(stock_order)
    ][
        [
            "date",
            "order_book_id",
            "ret_1m",
            "ret_fwd_1m",
            "mvel1",
            "feature_nonnull_raw",
            *needed_raw_cols,
        ]
    ].copy()
    model_rows = _median_impute_and_rank(model_rows, manifest, kept_feature_cols)
    arrays = fixed_panel_to_arrays(model_rows, kept_feature_cols, all_dates, stock_order)

    train_idx = np.arange(len(train_dates), dtype=int)
    raw_test_idx = np.arange(len(train_dates), len(all_dates), dtype=int)
    test_idx = raw_test_idx[arrays.mask[raw_test_idx].sum(axis=1) > 0]
    if len(test_idx) == 0:
        raise ValueError(f"{test_year}: no test months have any selected-stock forward returns.")
    split = Split(
        train_start_year=test_year - train_years,
        train_end_year=test_year - 1,
        test_year=test_year,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    if n_action_stocks == n_stocks:
        action_groups = selected.copy()
    else:
        action_groups = _sample_balanced_ff6(
            selected,
            n_total=n_action_stocks,
            seed=seed + 100_000 + test_year,
        )
    action_stock_order = action_groups["order_book_id"].astype(str).to_numpy()
    slot_lookup = {sid: i for i, sid in enumerate(stock_order)}
    action_slot_indices = np.asarray([slot_lookup[sid] for sid in action_stock_order], dtype=int)

    train_mask_counts = arrays.mask[split.train_idx].sum(axis=1)
    test_mask_counts = arrays.mask[split.test_idx].sum(axis=1)
    audit = {
        "test_year": test_year,
        "selection_date": selection_date,
        "n_train_months": len(train_dates),
        "n_test_months": len(test_idx),
        "n_raw_test_months": len(test_dates),
        "dropped_empty_test_months": int(len(raw_test_idx) - len(test_idx)),
        "n_stocks": n_stocks,
        "n_action_stocks": n_action_stocks,
        "n_features": len(kept_feature_cols),
        "n_dropped_features": len(dropped_feature_cols),
        "train_mask_min": int(train_mask_counts.min()) if len(train_mask_counts) else 0,
        "train_mask_max": int(train_mask_counts.max()) if len(train_mask_counts) else 0,
        "test_mask_min": int(test_mask_counts.min()) if len(test_mask_counts) else 0,
        "test_mask_max": int(test_mask_counts.max()) if len(test_mask_counts) else 0,
        "eligible_stocks": len(eligible),
        "ff6_selected_counts": _ff6_count_dict(selected),
        "ff6_action_counts": _ff6_count_dict(action_groups),
    }
    return FF6SplitData(
        split=split,
        selection_date=selection_date,
        arrays=arrays,
        stock_order=stock_order,
        stock_groups=selected.reset_index(drop=True),
        action_stock_order=action_stock_order,
        action_slot_indices=action_slot_indices,
        action_groups=action_groups.reset_index(drop=True),
        kept_feature_cols=kept_feature_cols,
        dropped_feature_cols=dropped_feature_cols,
        missingness_report=missingness,
        audit=audit,
    )


class FixedActionPortfolioEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        factors: np.ndarray,
        forward_returns: np.ndarray,
        mask: np.ndarray,
        dates: pd.DatetimeIndex,
        indices: np.ndarray,
        action_slot_indices: np.ndarray,
        lookback: int = 12,
        reward_mode: Literal["log_return", "differential_sharpe", "differential_ddr"] = "log_return",
        action_mode: Literal["long_short", "long_only"] = "long_short",
        long_only_temperature: float = 0.2,
        eta: float = 0.01,
    ) -> None:
        if spaces is None:
            raise ImportError("Install gymnasium before constructing FixedActionPortfolioEnv.")
        self.factors = factors.astype(np.float32)
        self.forward_returns = forward_returns.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.dates = dates
        self.indices = np.asarray(indices, dtype=int)
        self.action_slot_indices = np.asarray(action_slot_indices, dtype=int)
        self.lookback = lookback
        self.reward_mode = reward_mode
        self.action_mode = action_mode
        self.long_only_temperature = long_only_temperature
        self.eta = eta
        self.n_factors = factors.shape[1]
        self.n_actions = len(self.action_slot_indices)
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
        action_mask = self.mask[month_idx, self.action_slot_indices]
        action_returns = self.forward_returns[month_idx, self.action_slot_indices]
        if self.action_mode == "long_short":
            action_weights = normalize_long_short(action, action_mask)
        elif self.action_mode == "long_only":
            action_weights = normalize_long_only(
                action,
                action_mask,
                temperature=self.long_only_temperature,
            )
        else:
            raise ValueError(f"unknown action_mode: {self.action_mode}")
        full_weights = np.zeros(self.forward_returns.shape[1], dtype=np.float32)
        full_weights[self.action_slot_indices] = action_weights
        portfolio_return = float(np.dot(action_weights, action_returns))
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
            "weights": full_weights,
            "action_weights": action_weights,
            "gross_long": float(action_weights[action_weights > 0].sum()),
            "gross_short": float(action_weights[action_weights < 0].sum()),
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


def evaluate_raw_action_ensemble(models: list[Any], env: FixedActionPortfolioEnv) -> pd.DataFrame:
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        actions = []
        for model in models:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(np.asarray(action, dtype=np.float32))
        avg_action = np.mean(np.stack(actions, axis=0), axis=0)
        obs, reward, terminated, truncated, info = env.step(avg_action)
        done = terminated or truncated
        action_weights = np.asarray(info["action_weights"], dtype=np.float64)
        abs_weights = np.abs(action_weights)
        hhi = float(np.sum(action_weights**2))
        effective_n = float(1.0 / hhi) if hhi > 1e-12 else np.nan
        rows.append(
            {
                "date": info["date"],
                "portfolio_return": info["portfolio_return"],
                "reward": reward,
                "gross_long": info["gross_long"],
                "gross_short": info["gross_short"],
                "ensemble_members": len(models),
                "max_weight": float(action_weights.max()) if len(action_weights) else np.nan,
                "min_weight": float(action_weights.min()) if len(action_weights) else np.nan,
                "max_abs_weight": float(abs_weights.max()) if len(abs_weights) else np.nan,
                "hhi": hhi,
                "effective_n": effective_n,
            }
        )
    return pd.DataFrame(rows)
