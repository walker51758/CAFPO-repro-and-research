from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - notebook install cell handles this path.
    try:
        import gym
        from gym import spaces
    except ImportError:  # pragma: no cover
        gym = None
        spaces = None

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError:  # pragma: no cover - imported after optional install in notebook.
    BaseFeaturesExtractor = nn.Module


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "rqdata_output"
MODEL_PANEL_PATH = OUTPUT_DIR / "cafpo_82_model_ready_top200_yearly.parquet"
PREPANEL_PATH = OUTPUT_DIR / "cafpo_82_prepanel_allstocks.parquet"
MANIFEST_PATH = OUTPUT_DIR / "cafpo_82_feature_manifest.csv"
CAE_OUTPUT_DIR = ROOT.parent / "cae" / "outputs" / "paper_faithful"
CAE_FACTOR_PANEL_PATH = CAE_OUTPUT_DIR / "attribution" / "paper_cae_CA1_K5_factor_panel.parquet"


@dataclass(frozen=True)
class CafpoArrays:
    dates: pd.DatetimeIndex
    feature_cols: list[str]
    x: np.ndarray
    ret_current: np.ndarray
    ret_forward: np.ndarray
    market_cap: np.ndarray
    mask: np.ndarray
    stock_ids: np.ndarray


@dataclass(frozen=True)
class Split:
    train_start_year: int
    train_end_year: int
    test_year: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class PaperCafpoData:
    dates: pd.DatetimeIndex
    feature_cols: list[str]
    features_3d: np.ndarray
    returns_2d: np.ndarray
    mask_2d: np.ndarray
    row_features: np.ndarray
    row_returns: np.ndarray
    row_date_idx: np.ndarray
    row_stock_idx: np.ndarray
    managed_returns: np.ndarray


@dataclass(frozen=True)
class ExternalCaeFactorModel:
    factors: np.ndarray
    dates: pd.DatetimeIndex
    source_path: Path
    test_year: int
    architecture: str
    n_factors: int
    cae_mode: str = "external"


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_panel(
    panel_path: Path = MODEL_PANEL_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(panel_path)
    manifest = pd.read_csv(manifest_path)
    feature_cols = manifest["feature_safe"].tolist()
    df["date"] = pd.to_datetime(df["date"])
    df["next_date"] = pd.to_datetime(df["next_date"])
    return df.sort_values(["date", "order_book_id"]).reset_index(drop=True), feature_cols


def audit_model_panel(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    duplicated_keys = int(df.duplicated(["date", "order_book_id"]).sum())
    feature_values = df[feature_cols].to_numpy(dtype=np.float64)
    ret_forward = df["ret_fwd_1m"].to_numpy(dtype=np.float64)
    by_month = df.groupby("date").size()

    audit = {
        "rows": int(len(df)),
        "n_months": int(df["date"].nunique()),
        "date_start": df["date"].min(),
        "date_end": df["date"].max(),
        "n_features": len(feature_cols),
        "duplicated_date_stock_keys": duplicated_keys,
        "stocks_per_month_min": int(by_month.min()),
        "stocks_per_month_max": int(by_month.max()),
        "stocks_per_month_mean": float(by_month.mean()),
        "feature_nan_count": int(np.isnan(feature_values).sum()),
        "feature_inf_count": int(np.isinf(feature_values).sum()),
        "feature_min": float(np.nanmin(feature_values)),
        "feature_max": float(np.nanmax(feature_values)),
        "target_nan_count": int(df["ret_fwd_1m"].isna().sum()),
        "target_inf_count": int(np.isinf(ret_forward).sum()),
        "target_describe": df["ret_fwd_1m"].describe(
            percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
        ),
        "feature_nonnull_raw_describe": df["feature_nonnull_raw"].describe(),
    }

    assert len(feature_cols) >= 83, f"expected at least 83 features, got {len(feature_cols)}"
    assert duplicated_keys == 0, "date/order_book_id keys are not unique"
    assert audit["feature_nan_count"] == 0, "model features contain NaN"
    assert audit["feature_inf_count"] == 0, "model features contain +/-inf"
    assert audit["target_nan_count"] == 0, "target returns contain NaN"
    assert audit["target_inf_count"] == 0, "target returns contain +/-inf"
    assert audit["feature_min"] >= -1.000001, "rank-normalized features are below -1"
    assert audit["feature_max"] <= 1.000001, "rank-normalized features are above 1"
    return audit


def audit_raw_missingness(
    prepanel_path: Path = PREPANEL_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[pd.Series, pd.Series]:
    pre = pd.read_parquet(prepanel_path)
    pre["date"] = pd.to_datetime(pre["date"])
    manifest = pd.read_csv(manifest_path)
    if "lagged_raw_col" in manifest.columns and set(manifest["lagged_raw_col"]).issubset(pre.columns):
        raw_feature_cols = manifest["lagged_raw_col"].tolist()
        missing_index = manifest["feature_original"].tolist()
    else:
        raw_feature_cols = manifest["feature_original"].tolist()
        missing_index = raw_feature_cols
    raw_top = pre[(pre["top200_year_flag"] == 1) & pre["ret_fwd_1m"].notna()].copy()
    feature_missing_rate = raw_top[raw_feature_cols].isna().mean()
    feature_missing_rate.index = pd.Index(missing_index, name="feature")
    feature_missing_rate = feature_missing_rate.sort_values(ascending=False)
    monthly_nonnull = raw_top.groupby("date")["feature_nonnull_raw"].mean().sort_values()
    return feature_missing_rate, monthly_nonnull


def build_rolling_splits(
    dates: pd.Series | pd.DatetimeIndex,
    train_years: int = 10,
    test_years: int = 1,
) -> list[Split]:
    date_index = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    years = sorted(date_index.year.unique().tolist())
    splits: list[Split] = []
    for test_year in years:
        train_start = test_year - train_years
        train_end = test_year - 1
        if train_start < years[0]:
            continue
        test_end = test_year + test_years - 1
        train_mask = (date_index.year >= train_start) & (date_index.year <= train_end)
        test_mask = (date_index.year >= test_year) & (date_index.year <= test_end)
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        if len(train_idx) and len(test_idx):
            splits.append(
                Split(
                    train_start_year=train_start,
                    train_end_year=train_end,
                    test_year=test_year,
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
            )
    return splits


def panel_to_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    max_stocks: int = 200,
) -> CafpoArrays:
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    n_months = len(dates)
    n_features = len(feature_cols)
    x = np.zeros((n_months, max_stocks, n_features), dtype=np.float32)
    ret_current = np.zeros((n_months, max_stocks), dtype=np.float32)
    ret_forward = np.zeros((n_months, max_stocks), dtype=np.float32)
    market_cap = np.zeros((n_months, max_stocks), dtype=np.float32)
    mask = np.zeros((n_months, max_stocks), dtype=np.float32)
    stock_ids = np.full((n_months, max_stocks), "", dtype="<U32")

    for t, date in enumerate(dates):
        month_df = df[df["date"] == date].sort_values("order_book_id").head(max_stocks)
        n = len(month_df)
        x[t, :n] = month_df[feature_cols].to_numpy(dtype=np.float32)
        ret_current[t, :n] = month_df["ret_1m"].to_numpy(dtype=np.float32)
        ret_forward[t, :n] = month_df["ret_fwd_1m"].to_numpy(dtype=np.float32)
        market_cap[t, :n] = month_df["mvel1"].fillna(0.0).to_numpy(dtype=np.float32)
        mask[t, :n] = 1.0
        stock_ids[t, :n] = month_df["order_book_id"].astype(str).to_numpy()

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


def infer_external_cae_test_year(
    arrays: CafpoArrays,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
) -> int:
    if val_idx is not None and len(val_idx):
        return int(pd.DatetimeIndex(arrays.dates[np.asarray(val_idx, dtype=int)]).year.max() + 1)
    if train_idx is not None and len(train_idx):
        return int(pd.DatetimeIndex(arrays.dates[np.asarray(train_idx, dtype=int)]).year.max() + 1)
    return int(pd.DatetimeIndex(arrays.dates).year.max())


def load_external_cae_factors(
    target_dates: pd.DatetimeIndex,
    test_year: int,
    architecture: str = "CA1",
    n_factors: int = 5,
    output_dir: Path = CAE_OUTPUT_DIR,
    factor_panel_path: Path = CAE_FACTOR_PANEL_PATH,
) -> ExternalCaeFactorModel:
    target_dates = pd.DatetimeIndex(pd.to_datetime(target_dates))
    output_dir = Path(output_dir)
    factor_panel_path = Path(factor_panel_path)
    npz_path = output_dir / f"paper_cae_{architecture}_K{n_factors}_test_{test_year}.npz"

    if npz_path.exists():
        with np.load(npz_path) as payload:
            source_dates = pd.DatetimeIndex(pd.to_datetime(payload["dates"]))
            source_factors = payload["factors"].astype(np.float32)
        source = pd.DataFrame(source_factors, index=source_dates).sort_index()
        missing = target_dates.difference(source.index)
        if len(missing):
            unexpected = missing[missing >= source.index.min()]
            if len(unexpected):
                raise ValueError(
                    f"External CAE factor file {npz_path} is missing {len(unexpected)} target dates; "
                    f"first missing date is {unexpected[0]}."
                )
        factors = source.reindex(target_dates).fillna(0.0).to_numpy(dtype=np.float32)
        return ExternalCaeFactorModel(
            factors=factors,
            dates=target_dates,
            source_path=npz_path,
            test_year=int(test_year),
            architecture=architecture,
            n_factors=int(n_factors),
        )

    if not factor_panel_path.exists():
        raise FileNotFoundError(
            f"Could not find external CAE factors at {npz_path} or {factor_panel_path}."
        )

    panel = pd.read_parquet(factor_panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[
        panel["architecture"].eq(architecture)
        & panel["test_year"].eq(int(test_year))
    ].copy()
    factor_cols = [f"factor_{k}" for k in range(1, n_factors + 1)]
    missing_cols = [col for col in factor_cols if col not in panel.columns]
    if missing_cols:
        raise KeyError(f"External CAE factor panel is missing columns: {missing_cols}.")
    if panel.empty:
        raise ValueError(f"No external CAE factor rows for {architecture} K={n_factors}, test_year={test_year}.")
    if panel.duplicated("date").any():
        dup = panel.loc[panel.duplicated("date"), "date"].iloc[0]
        raise ValueError(f"External CAE factor panel has duplicate date for test_year={test_year}: {dup}.")

    source = panel.set_index("date")[factor_cols].sort_index()
    missing = target_dates.difference(source.index)
    if len(missing):
        unexpected = missing[missing >= source.index.min()]
        if len(unexpected):
            raise ValueError(
                f"External CAE factor panel {factor_panel_path} is missing {len(unexpected)} target dates; "
                f"first missing date is {unexpected[0]}."
            )
    factors = source.reindex(target_dates).fillna(0.0).to_numpy(dtype=np.float32)
    return ExternalCaeFactorModel(
        factors=factors,
        dates=target_dates,
        source_path=factor_panel_path,
        test_year=int(test_year),
        architecture=architecture,
        n_factors=int(n_factors),
    )


class ConditionalAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        max_stocks: int,
        n_factors: int = 5,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.beta_net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_factors),
        )
        self.factor_net = nn.Linear(max_stocks, n_factors)

    def forward(self, x: torch.Tensor, returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        beta = self.beta_net(x)
        factors = self.factor_net(returns)
        recon = torch.einsum("bnk,bk->bn", beta, factors)
        return recon, beta, factors


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = ((pred - target) ** 2) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


class PaperCafpoRowDataset(torch.utils.data.Dataset):
    def __init__(self, data: PaperCafpoData, row_idx: np.ndarray) -> None:
        self.features = torch.tensor(data.row_features[row_idx], dtype=torch.float32)
        self.returns = torch.tensor(data.row_returns[row_idx], dtype=torch.float32)
        self.date_idx = torch.tensor(data.row_date_idx[row_idx], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.returns)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.date_idx[idx], self.returns[idx]


class PaperConditionalAutoencoder(nn.Module):
    """Paper-faithful CAE: betas from characteristics, factors from managed returns."""

    def __init__(
        self,
        n_features: int,
        n_factor_inputs: int,
        n_factors: int = 5,
        architecture: Literal["CA1", "CA2"] = "CA1",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if architecture == "CA1":
            beta_layers: list[nn.Module] = [
                nn.Linear(n_features, 32),
                nn.ReLU(),
            ]
            if dropout > 0:
                beta_layers.append(nn.Dropout(dropout))
            beta_layers.append(nn.Linear(32, n_factors))
        elif architecture == "CA2":
            beta_layers = [
                nn.Linear(n_features, 32),
                nn.ReLU(),
            ]
            if dropout > 0:
                beta_layers.append(nn.Dropout(dropout))
            beta_layers.extend([nn.Linear(32, 16), nn.ReLU()])
            if dropout > 0:
                beta_layers.append(nn.Dropout(dropout))
            beta_layers.append(nn.Linear(16, n_factors))
        else:
            raise ValueError("architecture must be 'CA1' or 'CA2'.")
        self.architecture = architecture
        self.beta_net = nn.Sequential(*beta_layers)
        self.factor_net = nn.Linear(n_factor_inputs, n_factors, bias=False)

    def forward(
        self,
        features: torch.Tensor,
        date_idx: torch.Tensor,
        managed_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        beta = self.beta_net(features)
        factors_all = self.factor_net(managed_returns)
        factors = factors_all[date_idx]
        pred = torch.sum(beta * factors, dim=1)
        return pred, beta, factors_all


def prepare_paper_cafpo_data(
    arrays: CafpoArrays,
    include_constant: bool = True,
    ridge: float = 1e-6,
) -> PaperCafpoData:
    """Convert date x stock arrays to the managed-return CAE representation.

    For each month t, the factor-side input is
    (Z_t' Z_t + ridge I)^-1 Z_t' r_t, matching the managed portfolio return
    construction used by the validated CAE workflow.
    """
    features = arrays.x.astype(np.float32)
    feature_cols = list(arrays.feature_cols)
    if include_constant:
        const = np.ones((*features.shape[:2], 1), dtype=np.float32)
        features = np.concatenate([const, features], axis=2)
        feature_cols = ["x_const", *feature_cols]

    returns = arrays.ret_current.astype(np.float32)
    mask = (
        arrays.mask.astype(bool)
        & np.isfinite(returns)
        & np.isfinite(features).all(axis=2)
        & (arrays.stock_ids != "")
    )

    n_dates, n_stocks, n_features = features.shape
    managed = np.zeros((n_dates, n_features), dtype=np.float32)
    for date_idx in range(n_dates):
        valid = mask[date_idx]
        if valid.sum() <= n_features:
            continue
        z = features[date_idx, valid].astype(np.float64)
        r = returns[date_idx, valid].astype(np.float64)
        try:
            if ridge > 0:
                lhs = z.T @ z + float(ridge) * np.eye(n_features)
                rhs = z.T @ r
                x_t = np.linalg.solve(lhs, rhs)
            else:
                x_t, *_ = np.linalg.lstsq(z, r, rcond=None)
        except np.linalg.LinAlgError:
            x_t = np.linalg.pinv(z) @ r
        managed[date_idx] = np.where(np.isfinite(x_t), x_t, 0.0).astype(np.float32)

    flat_mask = mask.reshape(-1)
    date_grid = np.repeat(np.arange(n_dates, dtype=np.int64), n_stocks)
    stock_grid = np.tile(np.arange(n_stocks, dtype=np.int64), n_dates)
    return PaperCafpoData(
        dates=arrays.dates,
        feature_cols=feature_cols,
        features_3d=features,
        returns_2d=returns,
        mask_2d=mask.astype(np.float32),
        row_features=features.reshape(-1, n_features)[flat_mask],
        row_returns=returns.reshape(-1)[flat_mask],
        row_date_idx=date_grid[flat_mask],
        row_stock_idx=stock_grid[flat_mask],
        managed_returns=managed,
    )


def _paper_rows_for_dates(data: PaperCafpoData, date_idx: np.ndarray) -> np.ndarray:
    date_idx = np.asarray(date_idx, dtype=np.int64)
    return np.flatnonzero(np.isin(data.row_date_idx, date_idx))


def paper_cafpo_loss(
    model: PaperConditionalAutoencoder,
    data: PaperCafpoData,
    date_idx: np.ndarray,
    batch_size: int = 8192,
    device: str | None = None,
) -> float:
    if device is None:
        device = next(model.parameters()).device.type
    row_idx = _paper_rows_for_dates(data, date_idx)
    if len(row_idx) == 0:
        return np.nan
    ds = PaperCafpoRowDataset(data, row_idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    managed = torch.tensor(data.managed_returns, dtype=torch.float32, device=device)
    losses = []
    counts = []
    model.eval()
    with torch.no_grad():
        for xb, db, rb in loader:
            xb = xb.to(device)
            db = db.to(device)
            rb = rb.to(device)
            pred, _, _ = model(xb, db, managed)
            loss = (pred - rb) ** 2
            losses.append(float(loss.sum().detach().cpu()))
            counts.append(len(rb))
    return float(np.sum(losses) / max(np.sum(counts), 1))


def train_paper_cafpo_cae(
    arrays: CafpoArrays,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    n_factors: int = 5,
    architecture: Literal["CA1", "CA2"] = "CA1",
    dropout: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    batch_size: int = 8192,
    patience: int = 20,
    seed: int = 42,
    device: str | None = None,
    include_constant: bool = True,
    factor_ridge: float = 1e-6,
) -> tuple[PaperConditionalAutoencoder, pd.DataFrame]:
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare_paper_cafpo_data(arrays, include_constant=include_constant, ridge=factor_ridge)
    train_rows = _paper_rows_for_dates(data, train_idx)
    if len(train_rows) == 0:
        raise ValueError("No valid CAE training rows after applying the array mask.")

    model = PaperConditionalAutoencoder(
        n_features=data.row_features.shape[1],
        n_factor_inputs=data.managed_returns.shape[1],
        n_factors=n_factors,
        architecture=architecture,
        dropout=dropout,
    ).to(device)
    model.cafpo_paper_data = data
    model.cae_mode = "paper"
    model.include_constant = include_constant
    model.factor_ridge = factor_ridge

    train_ds = PaperCafpoRowDataset(data, train_rows)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    managed = torch.tensor(data.managed_returns, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = math.inf
    wait = 0
    history: list[dict[str, float | int | str]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        counts = []
        for xb, db, rb in train_loader:
            xb = xb.to(device)
            db = db.to(device)
            rb = rb.to(device)
            pred, _, _ = model(xb, db, managed)
            loss = torch.mean((pred - rb) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()) * len(rb))
            counts.append(len(rb))

        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "train_loss": float(np.sum(losses) / max(np.sum(counts), 1)),
            "cae_mode": "paper",
            "architecture": architecture,
        }
        if val_idx is not None and len(val_idx):
            val_loss = paper_cafpo_loss(model, data, val_idx, batch_size=batch_size, device=device)
            row["val_loss"] = val_loss
            if np.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                history.append(row)
                break
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def train_cae(
    arrays: CafpoArrays,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    n_factors: int = 5,
    hidden_dim: int = 32,
    dropout: float | None = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    batch_size: int | None = None,
    patience: int = 20,
    seed: int = 42,
    device: str | None = None,
    cae_mode: Literal["external", "paper", "legacy"] = "external",
    architecture: Literal["CA1", "CA2"] = "CA1",
    include_constant: bool = True,
    factor_ridge: float = 1e-6,
    external_test_year: int | None = None,
    external_output_dir: Path = CAE_OUTPUT_DIR,
    external_factor_panel_path: Path = CAE_FACTOR_PANEL_PATH,
) -> tuple[Any, pd.DataFrame]:
    if cae_mode == "external":
        test_year = (
            int(external_test_year)
            if external_test_year is not None
            else infer_external_cae_test_year(arrays, train_idx=train_idx, val_idx=val_idx)
        )
        model = load_external_cae_factors(
            target_dates=arrays.dates,
            test_year=test_year,
            architecture=architecture,
            n_factors=n_factors,
            output_dir=external_output_dir,
            factor_panel_path=external_factor_panel_path,
        )
        history = pd.DataFrame(
            [
                {
                    "epoch": 0,
                    "train_loss": np.nan,
                    "val_loss": np.nan,
                    "cae_mode": "external",
                    "architecture": architecture,
                    "test_year": test_year,
                    "source_path": str(model.source_path),
                }
            ]
        )
        return model, history

    if cae_mode == "paper":
        return train_paper_cafpo_cae(
            arrays=arrays,
            train_idx=train_idx,
            val_idx=val_idx,
            n_factors=n_factors,
            architecture=architecture,
            dropout=0.0 if dropout is None else dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=8192 if batch_size is None else batch_size,
            patience=patience,
            seed=seed,
            device=device,
            include_constant=include_constant,
            factor_ridge=factor_ridge,
        )
    if cae_mode != "legacy":
        raise ValueError("cae_mode must be 'external', 'paper' or 'legacy'.")

    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConditionalAutoencoder(
        n_features=arrays.x.shape[-1],
        max_stocks=arrays.x.shape[1],
        n_factors=n_factors,
        hidden_dim=hidden_dim,
        dropout=0.1 if dropout is None else dropout,
    ).to(device)

    train_ds = TensorDataset(
        torch.tensor(arrays.x[train_idx], dtype=torch.float32),
        torch.tensor(arrays.ret_current[train_idx], dtype=torch.float32),
        torch.tensor(arrays.mask[train_idx], dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=32 if batch_size is None else batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = math.inf
    wait = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, rb, mb in train_loader:
            xb = xb.to(device)
            rb = rb.to(device)
            mb = mb.to(device)
            pred, _, _ = model(xb, rb)
            loss = masked_mse(pred, rb, mb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        row: dict[str, float | int] = {"epoch": epoch, "train_loss": float(np.mean(train_losses))}
        if val_idx is not None and len(val_idx):
            val_loss = evaluate_cae_loss(model, arrays, val_idx, device=device)
            row["val_loss"] = val_loss
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                history.append(row)
                break
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def evaluate_cae_loss(
    model: Any,
    arrays: CafpoArrays,
    idx: np.ndarray,
    device: str | None = None,
) -> float:
    if isinstance(model, ExternalCaeFactorModel):
        return np.nan

    if isinstance(model, PaperConditionalAutoencoder):
        data = getattr(model, "cafpo_paper_data", None)
        if data is None:
            data = prepare_paper_cafpo_data(arrays)
            model.cafpo_paper_data = data
        return paper_cafpo_loss(model, data, idx, device=device)

    if device is None:
        device = next(model.parameters()).device.type
    model.eval()
    with torch.no_grad():
        x = torch.tensor(arrays.x[idx], dtype=torch.float32, device=device)
        r = torch.tensor(arrays.ret_current[idx], dtype=torch.float32, device=device)
        m = torch.tensor(arrays.mask[idx], dtype=torch.float32, device=device)
        pred, _, _ = model(x, r)
        return float(masked_mse(pred, r, m).detach().cpu())


def extract_cae_outputs(
    model: Any,
    arrays: CafpoArrays,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(model, ExternalCaeFactorModel):
        if len(model.dates) != len(arrays.dates) or not model.dates.equals(pd.DatetimeIndex(arrays.dates)):
            aligned = load_external_cae_factors(
                target_dates=arrays.dates,
                test_year=model.test_year,
                architecture=model.architecture,
                n_factors=model.n_factors,
            )
            factors = aligned.factors
        else:
            factors = model.factors
        betas = np.zeros((*arrays.x.shape[:2], factors.shape[1]), dtype=np.float32)
        recon = np.zeros(arrays.ret_current.shape, dtype=np.float32)
        return factors.astype(np.float32), betas, recon

    if isinstance(model, PaperConditionalAutoencoder):
        if device is None:
            device = next(model.parameters()).device.type
        data = getattr(model, "cafpo_paper_data", None)
        if data is None:
            data = prepare_paper_cafpo_data(
                arrays,
                include_constant=bool(getattr(model, "include_constant", True)),
                ridge=float(getattr(model, "factor_ridge", 1e-6)),
            )
            model.cafpo_paper_data = data
        model.eval()
        factors = np.zeros((len(data.dates), model.factor_net.out_features), dtype=np.float32)
        betas = np.zeros((*arrays.x.shape[:2], model.factor_net.out_features), dtype=np.float32)
        recon = np.zeros(arrays.ret_current.shape, dtype=np.float32)
        managed = torch.tensor(data.managed_returns, dtype=torch.float32, device=device)
        ds = PaperCafpoRowDataset(data, np.arange(len(data.row_returns), dtype=int))
        loader = DataLoader(ds, batch_size=8192, shuffle=False)
        row_cursor = 0
        with torch.no_grad():
            factors[:] = model.factor_net(managed).detach().cpu().numpy().astype(np.float32)
            for xb, db, _ in loader:
                batch_len = len(db)
                xb = xb.to(device)
                db = db.to(device)
                pred, beta, _ = model(xb, db, managed)
                rows = slice(row_cursor, row_cursor + batch_len)
                date_pos = data.row_date_idx[rows]
                stock_pos = data.row_stock_idx[rows]
                betas[date_pos, stock_pos] = beta.detach().cpu().numpy().astype(np.float32)
                recon[date_pos, stock_pos] = pred.detach().cpu().numpy().astype(np.float32)
                row_cursor += batch_len
        return factors, betas, recon

    if device is None:
        device = next(model.parameters()).device.type
    model.eval()
    betas = []
    factors = []
    recon = []
    with torch.no_grad():
        for start in range(0, len(arrays.dates), 64):
            stop = min(start + 64, len(arrays.dates))
            xb = torch.tensor(arrays.x[start:stop], dtype=torch.float32, device=device)
            rb = torch.tensor(arrays.ret_current[start:stop], dtype=torch.float32, device=device)
            pred, beta, factor = model(xb, rb)
            betas.append(beta.detach().cpu().numpy())
            factors.append(factor.detach().cpu().numpy())
            recon.append(pred.detach().cpu().numpy())
    return np.concatenate(factors), np.concatenate(betas), np.concatenate(recon)


def normalize_long_short(action: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask.astype(bool)
    weights = np.zeros_like(action, dtype=np.float32)
    if valid.sum() < 2:
        return weights

    raw = np.asarray(action, dtype=np.float64)
    raw_valid = raw[valid]
    long_raw = np.clip(raw_valid, 0.0, None)
    short_raw = np.clip(raw_valid, None, 0.0)

    if long_raw.sum() <= 1e-12 or abs(short_raw.sum()) <= 1e-12:
        order = np.argsort(raw_valid)
        half = max(1, len(order) // 2)
        short_idx = order[:half]
        long_idx = order[-half:]
        long_raw = np.zeros_like(raw_valid)
        short_raw = np.zeros_like(raw_valid)
        long_raw[long_idx] = np.maximum(raw_valid[long_idx] - np.median(raw_valid), 1e-6)
        short_raw[short_idx] = np.minimum(raw_valid[short_idx] - np.median(raw_valid), -1e-6)

    long_sum = long_raw.sum()
    short_sum = abs(short_raw.sum())
    if long_sum > 0:
        long_raw = long_raw / long_sum
    if short_sum > 0:
        short_raw = short_raw / short_sum

    weights[valid] = (long_raw + short_raw).astype(np.float32)
    return weights


class CafpoPortfolioEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        factors: np.ndarray,
        forward_returns: np.ndarray,
        mask: np.ndarray,
        dates: pd.DatetimeIndex,
        indices: np.ndarray,
        lookback: int = 12,
        reward_mode: Literal["log_return", "differential_sharpe", "differential_ddr"] = "log_return",
        eta: float = 0.01,
    ) -> None:
        if spaces is None:
            raise ImportError("Install gymnasium before constructing CafpoPortfolioEnv.")
        self.factors = factors.astype(np.float32)
        self.forward_returns = forward_returns.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.dates = dates
        self.indices = np.asarray(indices, dtype=int)
        self.lookback = lookback
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
        weights = normalize_long_short(action, self.mask[month_idx])
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
                reward = (self.dd2_ema * (portfolio_return - 0.5 * self.a_ema) - 0.5 * self.a_ema * portfolio_return**2) / denom
            self.a_ema += self.eta * (portfolio_return - self.a_ema)
            self.dd2_ema += self.eta * (downside - self.dd2_ema)
            return reward

        raise ValueError(f"unknown reward_mode: {self.reward_mode}")


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Any, features_dim: int = 64, lstm_hidden: int = 32) -> None:
        if BaseFeaturesExtractor is nn.Module:
            raise ImportError("Install stable-baselines3 before constructing LSTMFeaturesExtractor.")
        super().__init__(observation_space, features_dim=features_dim)
        n_factors = int(observation_space.shape[-1])
        self.lstm = nn.LSTM(input_size=n_factors, hidden_size=lstm_hidden, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(lstm_hidden, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(observations)
        return self.proj(hidden[-1])


def make_ppo_model(
    env: CafpoPortfolioEnv,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int = 64,
    batch_size: int = 32,
    gamma: float = 1.0,
    verbose: int = 0,
) -> Any:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is not installed. Run: pip install 'stable-baselines3[extra]' gymnasium"
        ) from exc

    policy_kwargs = {
        "features_extractor_class": LSTMFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 64, "lstm_hidden": 32},
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
    }
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=verbose,
    )


def make_ddpg_model(
    env: CafpoPortfolioEnv,
    seed: int = 42,
    learning_rate: float = 1e-4,
    buffer_size: int = 50_000,
    batch_size: int = 64,
    gamma: float = 1.0,
    verbose: int = 0,
) -> Any:
    try:
        from stable_baselines3 import DDPG
        from stable_baselines3.common.noise import NormalActionNoise
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is not installed. Run: pip install 'stable-baselines3[extra]' gymnasium"
        ) from exc

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions, dtype=np.float32),
        sigma=0.05 * np.ones(n_actions, dtype=np.float32),
    )
    policy_kwargs = {
        "features_extractor_class": LSTMFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 64, "lstm_hidden": 32},
        "net_arch": {"pi": [64, 64], "qf": [64, 64]},
    }
    return DDPG(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=verbose,
    )


def evaluate_sb3_policy(model: Any, env: CafpoPortfolioEnv, deterministic: bool = True) -> pd.DataFrame:
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append(
            {
                "date": info["date"],
                "portfolio_return": info["portfolio_return"],
                "reward": reward,
                "gross_long": info["gross_long"],
                "gross_short": info["gross_short"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_sb3_ensemble(
    models: list[Any],
    env: CafpoPortfolioEnv,
    deterministic: bool = True,
    ensemble_mode: Literal["mean_action", "mean_weight"] = "mean_action",
) -> pd.DataFrame:
    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        month_idx = int(env.indices[env.ptr])
        member_actions = []
        for model in models:
            action, _ = model.predict(obs, deterministic=deterministic)
            member_actions.append(np.asarray(action, dtype=np.float32))
        avg_action = np.mean(np.stack(member_actions, axis=0), axis=0)
        if ensemble_mode == "mean_action":
            obs, reward, terminated, truncated, info = env.step(avg_action)
        elif ensemble_mode == "mean_weight":
            avg_weights = np.mean(
                [normalize_long_short(action, env.mask[month_idx]) for action in member_actions],
                axis=0,
            )
            portfolio_return = float(np.dot(avg_weights, env.forward_returns[month_idx]))
            reward = env._reward(portfolio_return)
            env.ptr += 1
            terminated = env.ptr >= len(env.indices)
            truncated = False
            obs = (
                np.zeros((env.lookback, env.n_factors), dtype=np.float32)
                if terminated
                else env._obs_for_month(int(env.indices[env.ptr]))
            )
            info = {
                "date": env.dates[month_idx],
                "portfolio_return": portfolio_return,
                "gross_long": float(avg_weights[avg_weights > 0].sum()),
                "gross_short": float(avg_weights[avg_weights < 0].sum()),
            }
        else:
            raise ValueError("ensemble_mode must be 'mean_action' or 'mean_weight'.")
        done = terminated or truncated
        rows.append(
            {
                "date": info["date"],
                "portfolio_return": info["portfolio_return"],
                "reward": reward,
                "gross_long": info["gross_long"],
                "gross_short": info["gross_short"],
                "ensemble_members": len(models),
            }
        )
    return pd.DataFrame(rows)


def long_short_weights_from_signal(
    signal: np.ndarray,
    mask: np.ndarray,
    top_frac: float = 0.3,
    market_cap: np.ndarray | None = None,
) -> np.ndarray:
    valid_idx = np.flatnonzero(mask.astype(bool))
    weights = np.zeros_like(signal, dtype=np.float32)
    if len(valid_idx) < 2:
        return weights

    n_leg = max(1, int(len(valid_idx) * top_frac))
    ordered = valid_idx[np.argsort(signal[valid_idx])]
    short_idx = ordered[:n_leg]
    long_idx = ordered[-n_leg:]

    if market_cap is None:
        weights[long_idx] = 1.0 / len(long_idx)
        weights[short_idx] = -1.0 / len(short_idx)
    else:
        long_cap = np.maximum(market_cap[long_idx], 0.0)
        short_cap = np.maximum(market_cap[short_idx], 0.0)
        if long_cap.sum() <= 0:
            long_cap = np.ones_like(long_cap)
        if short_cap.sum() <= 0:
            short_cap = np.ones_like(short_cap)
        weights[long_idx] = long_cap / long_cap.sum()
        weights[short_idx] = -short_cap / short_cap.sum()
    return weights


def _historical_mean_signal_for_ids(
    arrays: CafpoArrays,
    ids: np.ndarray,
    train_idx: np.ndarray,
) -> np.ndarray:
    signal = np.zeros(len(ids), dtype=np.float32)
    for j, sid in enumerate(ids):
        values = []
        for t in train_idx:
            where = np.flatnonzero(arrays.stock_ids[t] == sid)
            if len(where):
                values.append(float(arrays.ret_forward[t, where[0]]))
        signal[j] = float(np.nanmean(values)) if values else 0.0
    return np.where(np.isfinite(signal), signal, 0.0).astype(np.float32)


def run_historical_sort_baseline(
    name: str,
    arrays: CafpoArrays,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    top_frac: float = 0.3,
    value_weighted: bool = False,
) -> pd.DataFrame:
    rows = []
    for idx in test_idx:
        valid = arrays.mask[idx].astype(bool)
        ids = arrays.stock_ids[idx, valid]
        signal_valid = _historical_mean_signal_for_ids(arrays, ids, train_idx)
        full_signal = np.zeros(arrays.mask.shape[1], dtype=np.float32)
        full_signal[valid] = signal_valid
        weights = long_short_weights_from_signal(
            full_signal,
            arrays.mask[idx],
            top_frac=top_frac,
            market_cap=arrays.market_cap[idx] if value_weighted else None,
        )
        rows.append(
            {
                "method": name,
                "date": arrays.dates[idx],
                "portfolio_return": float(np.dot(weights, arrays.ret_forward[idx])),
            }
        )
    return pd.DataFrame(rows)


def run_markowitz_baseline(
    arrays: CafpoArrays,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ridge: float = 1e-4,
) -> pd.DataFrame:
    rows = []
    for idx in test_idx:
        valid = arrays.mask[idx].astype(bool)
        ids = arrays.stock_ids[idx, valid]
        if len(ids) == 0:
            rows.append(
                {
                    "method": "Markowitz",
                    "date": arrays.dates[idx],
                    "portfolio_return": 0.0,
                }
            )
            continue
        returns = []
        for sid in ids:
            series = []
            for t in train_idx:
                where = np.flatnonzero(arrays.stock_ids[t] == sid)
                series.append(float(arrays.ret_forward[t, where[0]]) if len(where) else np.nan)
            returns.append(series)
        ret_matrix = np.asarray(returns, dtype=np.float64).T
        if ret_matrix.ndim != 2 or ret_matrix.shape[0] == 0 or ret_matrix.shape[1] == 0:
            rows.append(
                {
                    "method": "Markowitz",
                    "date": arrays.dates[idx],
                    "portfolio_return": 0.0,
                }
            )
            continue
        finite_counts = np.isfinite(ret_matrix).sum(axis=0)
        mu = np.divide(
            np.nansum(ret_matrix, axis=0),
            finite_counts,
            out=np.zeros(ret_matrix.shape[1], dtype=np.float64),
            where=finite_counts > 0,
        )
        filled = np.where(np.isfinite(ret_matrix), ret_matrix, mu)

        if filled.shape[0] >= 3 and filled.shape[1] >= 2:
            cov = LedoitWolf().fit(filled).covariance_
        else:
            cov = np.eye(len(ids))
        cov = cov + np.eye(cov.shape[0]) * ridge
        try:
            raw = np.linalg.solve(cov, mu)
        except np.linalg.LinAlgError:
            raw = np.linalg.pinv(cov) @ mu

        full_signal = np.zeros(arrays.mask.shape[1], dtype=np.float32)
        full_signal[valid] = raw.astype(np.float32)
        weights = normalize_long_short(full_signal, arrays.mask[idx])
        rows.append(
            {
                "method": "Markowitz",
                "date": arrays.dates[idx],
                "portfolio_return": float(np.dot(weights, arrays.ret_forward[idx])),
            }
        )
    return pd.DataFrame(rows)


def performance_summary(returns: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    rows = []
    for method, group in returns.groupby(method_col):
        r = group["portfolio_return"].to_numpy(dtype=np.float64)
        wealth = np.cumprod(1.0 + r)
        drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
        compound = wealth[-1] - 1.0 if len(wealth) else np.nan
        sharpe = np.nan if np.std(r) == 0 else np.mean(r) / np.std(r, ddof=1)
        max_drawdown = float(drawdown.min()) if len(drawdown) else np.nan
        sterling = np.nan if max_drawdown == 0 else np.mean(r) / abs(max_drawdown)
        rows.append(
            {
                "method": method,
                "months": len(r),
                "compound_return": compound,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "sterling_ratio": sterling,
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def save_cae_outputs(
    factors: np.ndarray,
    betas: np.ndarray,
    recon: np.ndarray,
    arrays: CafpoArrays,
    output_prefix: Path,
    cae_mode: str = "external_cae_factors",
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        factors=factors.astype(np.float32),
        betas=betas.astype(np.float32),
        reconstructed_returns=recon.astype(np.float32),
        dates=np.array([d.strftime("%Y-%m-%d") for d in arrays.dates], dtype="<U10"),
        stock_ids=arrays.stock_ids,
        feature_cols=np.array(arrays.feature_cols, dtype="<U64"),
        cae_mode=np.array(cae_mode, dtype="<U32"),
    )
