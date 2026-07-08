from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rqdatac as rq


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "rqdata_output"
OUTPUT_DIR.mkdir(exist_ok=True)

BASE_PANEL_PATH = OUTPUT_DIR / "cafpo_82_prepanel_allstocks.parquet"
BASPREAD_PATH = OUTPUT_DIR / "cafpo_baspread_monthly.parquet"
PARTS_DIR = OUTPUT_DIR / "baspread_monthly_parts"
PARTS_DIR.mkdir(exist_ok=True)
PROGRESS_PATH = OUTPUT_DIR / "baspread_progress.csv"

DEFAULT_SOURCE = os.getenv("RQDATA_BASPREAD_SOURCE", "daily").strip().lower()
if DEFAULT_SOURCE not in {"daily", "minute"}:
    raise ValueError("RQDATA_BASPREAD_SOURCE must be either 'daily' or 'minute'")

BATCH_SIZE = int(os.getenv("RQDATA_BASPREAD_BATCH_SIZE", "1200" if DEFAULT_SOURCE == "daily" else "200"))
MIN_DAILY_OBS = int(os.getenv("RQDATA_BASPREAD_MIN_OBS", "12"))


def batch_list(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


def load_month_universe() -> pd.DataFrame:
    if not BASE_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing base panel: {BASE_PANEL_PATH}")
    panel = pd.read_parquet(BASE_PANEL_PATH, columns=["date", "order_book_id"])
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def fetch_price_block(ids: list[str], month_start: pd.Timestamp, month_end: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    frequency = "1m" if DEFAULT_SOURCE == "minute" else "1d"
    fields = ["high", "low", "volume"]
    for chunk in batch_list(ids, BATCH_SIZE):
        df = rq.get_price(
            chunk,
            start_date=month_start,
            end_date=month_end,
            frequency=frequency,
            fields=fields,
            skip_suspended=False,
        )
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["high", "low", "volume"])
    return pd.concat(frames, axis=0)


def normalize_price_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame(columns=["order_book_id", "date", "high", "low", "volume"])

    frame = price_df.reset_index().copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    elif "datetime" in frame.columns:
        frame["date"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    else:
        raise KeyError("Price frame is missing both date and datetime index columns")

    cols = ["order_book_id", "date", "high", "low"]
    if "volume" in frame.columns:
        cols.append("volume")
    return frame[cols]


def corwin_schultz_monthly_spread(price_df: pd.DataFrame, ids: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=pd.Index(ids, name="order_book_id"), name="baspread")
    if price_df.empty:
        return out

    daily = normalize_price_frame(price_df)
    agg = {"high": "max", "low": "min"}
    if "volume" in daily.columns:
        agg["volume"] = "sum"
    daily = daily.groupby(["order_book_id", "date"], as_index=False).agg(agg)

    if "volume" in daily.columns:
        daily = daily.loc[daily["volume"].fillna(0) > 0].copy()
    daily = daily.loc[daily["high"].gt(0) & daily["low"].gt(0)].copy()
    if daily.empty:
        return out

    daily = daily.sort_values(["order_book_id", "date"]).copy()
    prev = daily.groupby("order_book_id", sort=False)[["high", "low"]].shift(1)

    valid = (
        prev["high"].notna()
        & prev["low"].notna()
        & daily["high"].notna()
        & daily["low"].notna()
        & (daily["high"] > 0)
        & (daily["low"] > 0)
        & (prev["high"] > 0)
        & (prev["low"] > 0)
        & (daily["high"] >= daily["low"])
        & (prev["high"] >= prev["low"])
    )
    if not valid.any():
        return out

    high = daily.loc[valid, "high"].to_numpy(dtype=float)
    low = daily.loc[valid, "low"].to_numpy(dtype=float)
    prev_high = prev.loc[valid, "high"].to_numpy(dtype=float)
    prev_low = prev.loc[valid, "low"].to_numpy(dtype=float)

    beta = np.log(high / low) ** 2 + np.log(prev_high / prev_low) ** 2
    beta = np.clip(beta, 0.0, None)
    gamma = np.log(np.maximum(high, prev_high) / np.minimum(low, prev_low)) ** 2
    gamma = np.clip(gamma, 0.0, None)

    const = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / const - np.sqrt(gamma / const)
    alpha = np.clip(alpha, -50.0, 50.0)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = np.where(np.isfinite(spread), np.maximum(spread, 0.0), np.nan)

    daily = daily.loc[valid, ["order_book_id"]].copy()
    daily["daily_baspread"] = spread

    monthly = daily.groupby("order_book_id")["daily_baspread"].agg(
        lambda s: s.dropna().mean() if s.notna().sum() >= MIN_DAILY_OBS else np.nan
    )
    return monthly.reindex(ids).rename("baspread")


def main() -> None:
    rq.init()
    panel = load_month_universe()
    month_dates = pd.DatetimeIndex(sorted(panel["date"].drop_duplicates()))

    progress_records: list[dict[str, object]] = []
    for idx, month_end in enumerate(month_dates, start=1):
        part_path = PARTS_DIR / f"{month_end:%Y%m}.parquet"
        if part_path.exists():
            existing_rows = len(pd.read_parquet(part_path, columns=["order_book_id"]))
            progress_records.append(
                {"date": month_end, "rows": existing_rows, "status": "skipped_existing"}
            )
            pd.DataFrame(progress_records).to_csv(PROGRESS_PATH, index=False, encoding="utf-8-sig")
            print(
                f"[{idx}/{len(month_dates)}] skipped existing {month_end.date()} rows={existing_rows}",
                flush=True,
            )
            continue

        ids = panel.loc[panel["date"] == month_end, "order_book_id"].drop_duplicates().tolist()
        month_start = month_end.replace(day=1)
        price_df = fetch_price_block(ids, month_start, month_end)
        baspread = corwin_schultz_monthly_spread(price_df, ids)
        month_frame = baspread.reset_index()
        month_frame["date"] = month_end
        month_frame = month_frame[["date", "order_book_id", "baspread"]]
        month_frame.to_parquet(part_path, index=False)
        progress_records.append({"date": month_end, "rows": len(month_frame), "status": "built"})
        pd.DataFrame(progress_records).to_csv(PROGRESS_PATH, index=False, encoding="utf-8-sig")
        print(f"[{idx}/{len(month_dates)}] built {month_end.date()} rows={len(month_frame)}", flush=True)

    part_files = sorted(PARTS_DIR.glob("*.parquet"))
    if not part_files:
        raise RuntimeError("No baspread parts were generated")

    baspread_panel = pd.concat((pd.read_parquet(path) for path in part_files), axis=0, ignore_index=True)
    baspread_panel["date"] = pd.to_datetime(baspread_panel["date"])
    baspread_panel = baspread_panel.sort_values(["date", "order_book_id"]).reset_index(drop=True)
    baspread_panel.to_parquet(BASPREAD_PATH, index=False)

    print(f"saved baspread panel to: {BASPREAD_PATH}", flush=True)
    print(f"saved progress to: {PROGRESS_PATH}", flush=True)


if __name__ == "__main__":
    main()
