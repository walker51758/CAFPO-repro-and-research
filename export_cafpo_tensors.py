from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "rqdata_output"
MODEL_TOP200_PATH = OUTPUT_DIR / "cafpo_82_model_ready_top200_yearly.parquet"
MANIFEST_PATH = OUTPUT_DIR / "cafpo_82_feature_manifest.csv"
NPZ_PATH = OUTPUT_DIR / "cafpo_82_top200_tensors.npz"
META_PATH = OUTPUT_DIR / "cafpo_82_top200_tensors_meta.json"


def main() -> None:
    df = pd.read_parquet(MODEL_TOP200_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    feature_cols = manifest["feature_safe"].tolist()
    ret_hist_cols = ["ret_1m"] + [f"ret_lag{lag:02d}" for lag in range(1, 13)]
    months = sorted(pd.to_datetime(df["date"].unique()).tolist())
    max_n = int(df.groupby("date").size().max())
    n_t = len(months)
    n_f = len(feature_cols)
    n_r = len(ret_hist_cols)

    x = np.zeros((n_t, max_n, n_f), dtype=np.float32)
    y = np.zeros((n_t, max_n), dtype=np.float32)
    ret_hist = np.zeros((n_t, max_n, n_r), dtype=np.float32)
    mask = np.zeros((n_t, max_n), dtype=np.int8)
    mcap = np.zeros((n_t, max_n), dtype=np.float32)
    stock_ids = np.full((n_t, max_n), "", dtype="<U16")

    for t, month in enumerate(months):
        month_df = df[df["date"] == month].sort_values("order_book_id").reset_index(drop=True)
        n = len(month_df)
        mask[t, :n] = 1
        x[t, :n, :] = month_df[feature_cols].to_numpy(dtype=np.float32)
        y[t, :n] = month_df["ret_fwd_1m"].to_numpy(dtype=np.float32)
        ret_hist[t, :n, :] = month_df[ret_hist_cols].to_numpy(dtype=np.float32)
        mcap[t, :n] = month_df["mvel1"].to_numpy(dtype=np.float32)
        stock_ids[t, :n] = month_df["order_book_id"].astype(str).to_numpy()

    np.savez_compressed(
        NPZ_PATH,
        X=x,
        y=y,
        ret_hist=ret_hist,
        mask=mask,
        market_cap=mcap,
        stock_ids=stock_ids,
        dates=np.array([d.strftime("%Y-%m-%d") for d in months], dtype="<U10"),
        feature_cols=np.array(feature_cols, dtype="<U64"),
        ret_hist_cols=np.array(ret_hist_cols, dtype="<U16"),
    )

    meta = {
        "n_months": n_t,
        "max_stocks_per_month": max_n,
        "n_features": n_f,
        "n_return_history_features": n_r,
        "feature_cols": feature_cols,
        "ret_hist_cols": ret_hist_cols,
        "date_start": months[0].strftime("%Y-%m-%d"),
        "date_end": months[-1].strftime("%Y-%m-%d"),
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved tensor package to: {NPZ_PATH}", flush=True)
    print(f"saved tensor metadata to: {META_PATH}", flush=True)


if __name__ == "__main__":
    main()
