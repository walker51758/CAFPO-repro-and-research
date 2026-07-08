from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "rqdata_output"
GKX_CHARACTERISTICS_PATH = ROOT / "Gu_Kelly_Xiu_94_Firm_Characteristics.xlsx"

FEATURE_PANEL_PATH = OUTPUT_DIR / "rqdata_82_characteristics_monthly.parquet"
RAW_PANEL_PATH = OUTPUT_DIR / "rqdata_direct_easy_raw_monthly.parquet"
BASPREAD_PANEL_PATH = OUTPUT_DIR / "cafpo_baspread_monthly.parquet"

PREP_ALL_PATH = OUTPUT_DIR / "cafpo_82_prepanel_allstocks.parquet"
MODEL_TOP200_PATH = OUTPUT_DIR / "cafpo_82_model_ready_top200_yearly.parquet"
FEATURE_MANIFEST_PATH = OUTPUT_DIR / "cafpo_82_feature_manifest.csv"
SPLIT_PATH = OUTPUT_DIR / "cafpo_rolling_splits_10y_1y.csv"
MISSING12_PATH = OUTPUT_DIR / "cafpo_missing_12_features.csv"

MISSING12 = [
    ("aeavol", "需要公告日前后收益窗口与事件对齐，当前流程未做事件研究框架"),
    ("ear", "需要公告日前后收益窗口与事件对齐，当前流程未做事件研究框架"),
    ("rsup", "需要季度单产品/单业务收入惊喜类口径，米筐现成字段不足"),
    ("pricedelay", "需要更完整的周/月回归框架与市场收益构造"),
    ("orgcap", "需要长期资本化 SG&A/R&D 存量，口径依赖更长历史和研发数据"),
    ("sin", "需要美股式行业罪恶股映射，A股需额外自建行业标签"),
    ("secured", "需要担保债务明细字段，米筐现有可得性不足"),
    ("securedind", "依赖 secured 原始字段"),
    ("rd", "研发费用历史完整性不足"),
    ("rd mve", "依赖研发费用历史完整性不足"),
    ("rd sale", "依赖研发费用历史完整性不足"),
]


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return f"x_{cleaned}"


def normalize_feature_key(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(name).lower())


def load_gkx_frequency_map(path: Path = GKX_CHARACTERISTICS_PATH) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing GKX characteristic dictionary: {path}")
    gkx = pd.read_excel(path, sheet_name="Characteristics_94")
    frequency_map: dict[str, str] = {}
    for _, row in gkx.iterrows():
        frequency = str(row["Frequency"]).strip()
        for col in ("Appendix Acronym", "Dataset Column Name"):
            key = normalize_feature_key(row[col])
            if key:
                frequency_map[key] = frequency
    return frequency_map


def feature_frequency(feature: str, frequency_map: dict[str, str]) -> str:
    key = normalize_feature_key(feature)
    if key not in frequency_map:
        raise KeyError(f"Feature {feature!r} is not found in the GKX 94-characteristic dictionary.")
    return frequency_map[key]


def feature_lag_months(feature: str, frequency_map: dict[str, str]) -> int:
    frequency = feature_frequency(feature, frequency_map)
    lag_by_frequency = {"Monthly": 1, "Quarterly": 4, "Annual": 6}
    if frequency not in lag_by_frequency:
        raise ValueError(f"Unsupported GKX frequency for {feature!r}: {frequency!r}")
    return lag_by_frequency[frequency]


def build_feature_manifest(feature_cols: list[str], frequency_map: dict[str, str]) -> pd.DataFrame:
    records = []
    for col in feature_cols:
        safe = safe_name(col)
        frequency = feature_frequency(col, frequency_map)
        records.append(
            {
                "feature_original": col,
                "feature_safe": safe,
                "gkx_frequency": frequency,
                "lag_months": feature_lag_months(col, frequency_map),
                "lagged_raw_col": f"lagged_{safe}",
                "source": "rqdata_82_characteristics_monthly",
            }
        )
    return pd.DataFrame(records)


def add_return_columns(panel: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    merged = panel.merge(
        raw[["date", "order_book_id", "close_month_end", "mvel1"]],
        on=["date", "order_book_id"],
        how="left",
        suffixes=("", "_raw"),
    )
    merged = merged.sort_values(["order_book_id", "date"]).copy()
    grouped = merged.groupby("order_book_id", group_keys=False)

    merged["ret_1m"] = grouped["close_month_end"].pct_change(1)
    merged["ret_fwd_1m"] = grouped["close_month_end"].shift(-1) / merged["close_month_end"] - 1.0
    merged["log_ret_fwd_1m"] = np.log1p(merged["ret_fwd_1m"])
    merged["next_date"] = grouped["date"].shift(-1)

    for lag in range(1, 13):
        merged[f"ret_lag{lag:02d}"] = grouped["ret_1m"].shift(lag)

    return merged


def assign_top200_year_flag(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["year"] = df["date"].dt.year
    first_month_rows = (
        df.sort_values(["year", "date"])
        .groupby("year", as_index=False)
        .head(1)[["year", "date"]]
        .rename(columns={"date": "first_date_of_year"})
    )
    jan_panel = df.merge(first_month_rows, on="year", how="left")
    jan_panel = jan_panel[jan_panel["date"] == jan_panel["first_date_of_year"]].copy()
    jan_panel["mvel_rank"] = jan_panel.groupby("year")["mvel1"].rank(method="first", ascending=False)
    top200 = jan_panel.loc[jan_panel["mvel_rank"] <= 200, ["year", "order_book_id"]].copy()
    top200["top200_year_flag"] = 1
    df = df.merge(top200, on=["year", "order_book_id"], how="left")
    df["top200_year_flag"] = df["top200_year_flag"].fillna(0).astype(int)
    return df


def add_lagged_feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    frequency_map: dict[str, str],
) -> pd.DataFrame:
    out = df.sort_values(["order_book_id", "date"]).copy()
    grouped = out.groupby("order_book_id", group_keys=False)
    for col in feature_cols:
        out[f"lagged_{safe_name(col)}"] = grouped[col].shift(feature_lag_months(col, frequency_map))
    return out


def cross_section_median_impute(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    def fill_with_month_median(series: pd.Series) -> pd.Series:
        if series.notna().sum() == 0:
            return series
        return series.fillna(series.median())

    for col in feature_cols:
        out[col] = out.groupby("date")[col].transform(fill_with_month_median)
    return out


def rank_normalize_monthly(
    df: pd.DataFrame,
    feature_cols: list[str],
    source_cols: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if source_cols is None:
        source_cols = feature_cols

    def rank_to_unit(series: pd.Series) -> pd.Series:
        valid = series.notna()
        if valid.sum() == 0:
            return pd.Series(np.nan, index=series.index)
        if valid.sum() == 1:
            result = pd.Series(np.nan, index=series.index)
            result.loc[valid] = 0.0
            return result
        ranks = series.loc[valid].rank(method="average")
        scaled = 2.0 * ((ranks - 1.0) / (valid.sum() - 1.0)) - 1.0
        result = pd.Series(np.nan, index=series.index)
        result.loc[valid] = scaled
        return result

    for col, source_col in zip(feature_cols, source_cols, strict=True):
        out[safe_name(col)] = out.groupby("date")[source_col].transform(rank_to_unit)
    return out


def build_rolling_splits(dates: pd.Series) -> pd.DataFrame:
    years = sorted(pd.Index(dates.dt.year.unique()).tolist())
    records: list[dict[str, int]] = []
    for test_year in years:
        train_start = test_year - 10
        train_end = test_year - 1
        if train_start < years[0]:
            continue
        records.append(
            {
                "train_start_year": train_start,
                "train_end_year": train_end,
                "test_year": test_year,
            }
        )
    return pd.DataFrame(records)


def load_or_build_full_panel() -> tuple[pd.DataFrame, list[str]]:
    if FEATURE_PANEL_PATH.exists() and RAW_PANEL_PATH.exists():
        panel = pd.read_parquet(FEATURE_PANEL_PATH)
        raw = pd.read_parquet(RAW_PANEL_PATH)
        panel["date"] = pd.to_datetime(panel["date"])
        raw["date"] = pd.to_datetime(raw["date"])
        feature_cols = [col for col in panel.columns if col not in {"date", "order_book_id"}]
        if BASPREAD_PANEL_PATH.exists():
            baspread = pd.read_parquet(BASPREAD_PANEL_PATH)
            baspread["date"] = pd.to_datetime(baspread["date"])
            if "baspread" not in panel.columns:
                panel = panel.merge(baspread, on=["date", "order_book_id"], how="left")
            if "baspread" not in feature_cols:
                feature_cols = [*feature_cols, "baspread"]
        elif "baspread" in panel.columns and "baspread" not in feature_cols:
            feature_cols = [*feature_cols, "baspread"]
        full = add_return_columns(panel, raw)
        full = assign_top200_year_flag(full)
        return full, feature_cols

    if PREP_ALL_PATH.exists() and FEATURE_MANIFEST_PATH.exists():
        full = pd.read_parquet(PREP_ALL_PATH)
        full["date"] = pd.to_datetime(full["date"])
        if "next_date" in full.columns:
            full["next_date"] = pd.to_datetime(full["next_date"])
        manifest = pd.read_csv(FEATURE_MANIFEST_PATH)
        feature_cols = manifest["feature_original"].tolist()
        if BASPREAD_PANEL_PATH.exists():
            baspread = pd.read_parquet(BASPREAD_PANEL_PATH)
            baspread["date"] = pd.to_datetime(baspread["date"])
            if "baspread" not in full.columns:
                full = full.merge(baspread, on=["date", "order_book_id"], how="left")
            if "baspread" not in feature_cols:
                feature_cols = [*feature_cols, "baspread"]
        elif "baspread" in full.columns and "baspread" not in feature_cols:
            feature_cols = [*feature_cols, "baspread"]
        missing_cols = [col for col in feature_cols if col not in full.columns]
        if missing_cols:
            raise FileNotFoundError(f"prepanel is missing raw feature columns: {missing_cols[:5]}")
        return full, feature_cols

    raise FileNotFoundError(
        "Cannot build CAFPO panel. Need either upstream RQData parquet files "
        "or an existing cafpo_82_prepanel_allstocks.parquet plus feature manifest."
    )


def main() -> None:
    full, feature_cols = load_or_build_full_panel()
    frequency_map = load_gkx_frequency_map()
    manifest = build_feature_manifest(feature_cols, frequency_map)
    manifest.to_csv(FEATURE_MANIFEST_PATH, index=False, encoding="utf-8-sig")

    full = add_lagged_feature_columns(full, feature_cols, frequency_map)
    lagged_feature_cols = [f"lagged_{safe_name(col)}" for col in feature_cols]
    full["feature_nonnull_raw"] = full[lagged_feature_cols].notna().sum(axis=1)
    full.to_parquet(PREP_ALL_PATH, index=False)

    top200 = full[full["top200_year_flag"] == 1].copy()
    top200 = top200[top200["ret_fwd_1m"].notna()].copy()
    top200 = cross_section_median_impute(top200, lagged_feature_cols)
    top200 = rank_normalize_monthly(top200, feature_cols, source_cols=lagged_feature_cols)

    safe_feature_cols = [safe_name(col) for col in feature_cols]
    top200[safe_feature_cols] = top200[safe_feature_cols].fillna(0.0)
    ret_history_cols = ["ret_1m"] + [f"ret_lag{lag:02d}" for lag in range(1, 13)]
    top200[ret_history_cols] = top200[ret_history_cols].fillna(0.0)
    ordered_cols = (
        ["date", "next_date", "year", "order_book_id", "top200_year_flag", "mvel1", "close_month_end"]
        + ["ret_1m", "ret_fwd_1m", "log_ret_fwd_1m"]
        + [f"ret_lag{lag:02d}" for lag in range(1, 13)]
        + ["feature_nonnull_raw"]
        + safe_feature_cols
    )
    top200[ordered_cols].to_parquet(MODEL_TOP200_PATH, index=False)

    splits = build_rolling_splits(top200["date"])
    splits.to_csv(SPLIT_PATH, index=False, encoding="utf-8-sig")

    pd.DataFrame(MISSING12, columns=["feature", "reason"]).to_csv(
        MISSING12_PATH, index=False, encoding="utf-8-sig"
    )

    print(f"saved all-stock prepanel to: {PREP_ALL_PATH}", flush=True)
    print(f"saved top200 model-ready panel to: {MODEL_TOP200_PATH}", flush=True)
    print(f"saved feature manifest to: {FEATURE_MANIFEST_PATH}", flush=True)
    print(f"saved rolling splits to: {SPLIT_PATH}", flush=True)
    print(f"saved missing-feature list to: {MISSING12_PATH}", flush=True)


if __name__ == "__main__":
    main()
