from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import rqdatac as rq


START_DATE = "2007-01-01"
END_DATE = "2026-05-31"
BATCH_SIZE = 1200

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "rqdata_output"
OUTPUT_DIR.mkdir(exist_ok=True)
PARTS_DIR = OUTPUT_DIR / "monthly_parts"
PARTS_DIR.mkdir(exist_ok=True)
PROGRESS_PATH = OUTPUT_DIR / "progress.csv"


DIRECT_FACTOR_MAP = {
    "bm": "book_to_market_ratio_lf",
    "cashdebt": "ocf_to_debt_ttm",
    "cfp": "cfp_ratio_ttm",
    "currat": "current_ratio_lf",
    "dy": "dividend_yield_ttm",
    "ep": "ep_ratio_ttm",
    "lev": "book_leverage_lf",
    "mvel1": "market_cap_3",
    "quick": "quick_ratio_lf",
    "roaq": "return_on_asset_ttm",
    "roeq": "return_on_equity_ttm",
    "sp": "sp_ratio_ttm",
}

RAW_FACTOR_FIELDS = [
    "market_cap_3",
    "ocf_to_debt_ttm",
    "book_to_market_ratio_lf",
    "current_ratio_lf",
    "quick_ratio_lf",
    "dividend_yield_ttm",
    "ep_ratio_ttm",
    "book_leverage_lf",
    "return_on_asset_ttm",
    "return_on_equity_ttm",
    "sp_ratio_ttm",
    "cfp_ratio_ttm",
    "cash_equivalent_mrq_0",
    "total_assets_mrq_0",
    "total_assets_mrq_4",
    "inventory_mrq_0",
    "inventory_mrq_4",
    "net_accts_receivable_mrq_0",
    "operating_revenue_ttm_0",
    "operating_revenue_ttm_4",
    "total_fixed_assets_mrq_0",
    "gross_profit_ttm_0",
]

RISK_FIELDS = ["beta"]


def batch_list(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = numer / denom
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def get_month_end_dates(start_date: str, end_date: str) -> list[pd.Timestamp]:
    trading_dates = pd.to_datetime(rq.get_trading_dates(start_date, end_date))
    month_end_dates = (
        pd.Series(trading_dates)
        .groupby(pd.Series(trading_dates).dt.to_period("M"))
        .max()
        .tolist()
    )
    return [pd.Timestamp(x) for x in month_end_dates]


def build_universe_master() -> pd.DataFrame:
    instruments = rq.all_instruments("CS").copy()
    instruments["listed_date"] = pd.to_datetime(instruments["listed_date"], errors="coerce")
    instruments["de_listed_date_raw"] = instruments["de_listed_date"].astype(str)
    instruments["de_listed_date_cmp"] = instruments["de_listed_date_raw"].replace(
        {"0000-00-00": "2099-12-31", "2999-12-31": "2099-12-31"}
    )
    return instruments


def eligible_ids(instruments: pd.DataFrame, as_of_date: pd.Timestamp) -> list[str]:
    cutoff = str(as_of_date.date())
    subset = instruments[
        (instruments["listed_date"] <= as_of_date) & (instruments["de_listed_date_cmp"] >= cutoff)
    ]
    return subset["order_book_id"].tolist()


def empty_month_frame(ids: list[str], date: pd.Timestamp) -> pd.DataFrame:
    frame = pd.DataFrame(index=pd.Index(ids, name="order_book_id"))
    frame["date"] = date
    return frame


def fetch_factor_block(ids: list[str], date: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in batch_list(ids, BATCH_SIZE):
        df = rq.get_factor(chunk, RAW_FACTOR_FIELDS, start_date=date, end_date=date)
        if isinstance(df.index, pd.MultiIndex):
            if "date" in df.index.names:
                df = df.xs(date, level="date")
            else:
                df.index = df.index.get_level_values(0)
        df.index.name = "order_book_id"
        frames.append(df)
    factor_df = pd.concat(frames, axis=0) if frames else pd.DataFrame(index=pd.Index([], name="order_book_id"))
    return factor_df.reindex(ids)


def fetch_risk_block(ids: list[str], date: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in batch_list(ids, BATCH_SIZE):
        df = rq.get_factor_exposure(chunk, start_date=date, end_date=date, factors=RISK_FIELDS)
        if df is None or df.empty:
            tmp = pd.DataFrame(index=pd.Index(chunk, name="order_book_id"), columns=RISK_FIELDS)
        else:
            if "date" in df.index.names:
                df = df.xs(date, level="date")
            else:
                df.index = df.index.get_level_values(0)
            df.index.name = "order_book_id"
            tmp = df
        frames.append(tmp)
    risk_df = pd.concat(frames, axis=0) if frames else pd.DataFrame(index=pd.Index([], name="order_book_id"))
    return risk_df.reindex(ids)


def fetch_turnover_block(ids: list[str], date: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in batch_list(ids, BATCH_SIZE):
        df = rq.get_turnover_rate(chunk, start_date=date, end_date=date)
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values(0)
        df.index.name = "order_book_id"
        frames.append(df)
    turnover_df = pd.concat(frames, axis=0) if frames else pd.DataFrame(index=pd.Index([], name="order_book_id"))
    return turnover_df.reindex(ids)


def fetch_shares_block(ids: list[str], date: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in batch_list(ids, BATCH_SIZE):
        df = rq.get_shares(chunk, start_date=date, end_date=date, fields=["total", "circulation_a", "free_circulation"])
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values(0)
        df.index.name = "order_book_id"
        frames.append(df)
    shares_df = pd.concat(frames, axis=0) if frames else pd.DataFrame(index=pd.Index([], name="order_book_id"))
    return shares_df.reindex(ids)


def fetch_price_block(ids: list[str], month_start: pd.Timestamp, month_end: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in batch_list(ids, BATCH_SIZE):
        df = rq.get_price(
            chunk,
            start_date=month_start,
            end_date=month_end,
            frequency="1d",
            fields=["close", "total_turnover", "volume"],
        )
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["close", "total_turnover", "volume"])
    return pd.concat(frames, axis=0)


def fetch_staff_block(ids: list[str], month_end: pd.Timestamp) -> pd.DataFrame:
    start_date = month_end - pd.Timedelta(days=450)
    frames: list[pd.DataFrame] = []
    for chunk in batch_list(ids, BATCH_SIZE):
        df = rq.get_staff_count(chunk, start_date=start_date, end_date=month_end)
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["staff_count_latest"])
    staff = pd.concat(frames, axis=0).reset_index()
    staff = staff.sort_values(["order_book_id", "info_date", "end_date"])
    total_col = "total_staff" if "total_staff" in staff.columns else "staff_count"
    latest = staff.groupby("order_book_id").tail(1).set_index("order_book_id")
    return latest[[total_col]].rename(columns={total_col: "staff_count_latest"})


def fetch_convertible_stock_set(month_end: pd.Timestamp) -> set[str]:
    df = rq.convertible.all_instruments(date=month_end)
    if df is None or df.empty:
        return set()
    active = df[df["stock_code"].notna()].copy()
    return set(active["stock_code"].astype(str))


def compute_monthly_price_stats(price_df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=pd.Index(ids, name="order_book_id"))
    if price_df.empty:
        out["close_month_end"] = np.nan
        out["dolvol"] = np.nan
        out["maxret"] = np.nan
        out["retvol"] = np.nan
        out["zerotrade"] = np.nan
        return out

    daily = price_df.reset_index().sort_values(["order_book_id", "date"])
    daily["daily_ret"] = daily.groupby("order_book_id")["close"].pct_change()
    grouped = daily.groupby("order_book_id")
    stats = pd.DataFrame(
        {
            "close_month_end": grouped["close"].last(),
            "dolvol": np.log1p(grouped["total_turnover"].mean()),
            "maxret": grouped["daily_ret"].max(),
            "retvol": grouped["daily_ret"].std(),
            "zerotrade": grouped["volume"].apply(lambda x: float((x.fillna(0) == 0).mean())),
        }
    )
    return stats.reindex(ids)


def compute_month_frame(
    ids: list[str],
    month_end: pd.Timestamp,
    factor_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    shares_df: pd.DataFrame,
    price_stats: pd.DataFrame,
    staff_df: pd.DataFrame,
    convertible_stock_set: set[str],
) -> pd.DataFrame:
    frame = empty_month_frame(ids, month_end)
    frame = frame.join(factor_df)
    frame = frame.join(risk_df)
    frame = frame.join(turnover_df.add_prefix("turnover_"))
    frame = frame.join(shares_df.add_prefix("shares_"))
    frame = frame.join(price_stats)
    frame = frame.join(staff_df)

    for gu_name, rq_name in DIRECT_FACTOR_MAP.items():
        frame[gu_name] = frame[rq_name]

    frame["beta"] = frame["beta"]
    frame["betasq"] = frame["beta"] ** 2
    frame["turn"] = frame["turnover_month"]
    frame["cash"] = safe_div(frame["cash_equivalent_mrq_0"], frame["total_assets_mrq_0"])
    frame["agr"] = safe_div(frame["total_assets_mrq_0"], frame["total_assets_mrq_4"]) - 1.0
    frame["chinv"] = safe_div(frame["inventory_mrq_0"], frame["total_assets_mrq_0"]) - safe_div(
        frame["inventory_mrq_4"], frame["total_assets_mrq_4"]
    )
    frame["gma"] = safe_div(frame["gross_profit_ttm_0"], frame["total_assets_mrq_0"])
    frame["saleinv"] = safe_div(frame["operating_revenue_ttm_0"], frame["inventory_mrq_0"])
    frame["salerec"] = safe_div(frame["operating_revenue_ttm_0"], frame["net_accts_receivable_mrq_0"])
    frame["sgr"] = safe_div(frame["operating_revenue_ttm_0"], frame["operating_revenue_ttm_4"]) - 1.0
    tang_num = (
        frame["cash_equivalent_mrq_0"]
        + 0.715 * frame["net_accts_receivable_mrq_0"]
        + 0.547 * frame["inventory_mrq_0"]
        + 0.535 * frame["total_fixed_assets_mrq_0"]
    )
    frame["tang"] = safe_div(tang_num, frame["total_assets_mrq_0"])
    frame["convind"] = frame.index.to_series().isin(convertible_stock_set).astype(float)
    return frame


def finalize_panel(raw_monthly: pd.DataFrame) -> pd.DataFrame:
    panel = raw_monthly.sort_values(["order_book_id", "date"]).copy()
    grouped = panel.groupby("order_book_id", group_keys=False)

    panel["mom1m"] = grouped["close_month_end"].pct_change(1)
    panel["mom6m"] = grouped["close_month_end"].pct_change(6)
    panel["mom12m"] = grouped["close_month_end"].pct_change(12)
    panel["mom36m"] = grouped["close_month_end"].pct_change(36)
    prior_6m_mom = grouped["close_month_end"].transform(lambda s: s.shift(6) / s.shift(12) - 1.0)
    panel["chmom"] = panel["mom6m"] - prior_6m_mom
    panel["std_dolvol"] = grouped["dolvol"].rolling(12, min_periods=6).std().reset_index(level=0, drop=True)
    panel["std_turn"] = grouped["turn"].rolling(12, min_periods=6).std().reset_index(level=0, drop=True)
    panel["chcsho"] = grouped["shares_total"].pct_change(12)
    panel["hire"] = grouped["staff_count_latest"].pct_change(12)

    feature_cols = [
        "beta",
        "betasq",
        "bm",
        "cash",
        "cashdebt",
        "cfp",
        "currat",
        "dy",
        "ep",
        "lev",
        "mvel1",
        "quick",
        "roaq",
        "roeq",
        "sp",
        "turn",
        "agr",
        "chcsho",
        "chinv",
        "convind",
        "dolvol",
        "gma",
        "hire",
        "maxret",
        "mom1m",
        "mom6m",
        "mom12m",
        "mom36m",
        "chmom",
        "retvol",
        "saleinv",
        "salerec",
        "sgr",
        "std_dolvol",
        "std_turn",
        "tang",
        "zerotrade",
    ]

    out_cols = ["date", "order_book_id"] + feature_cols
    return panel.reset_index()[out_cols]


def build_feature_catalog() -> pd.DataFrame:
    records = [
        ("beta", "direct", "rq.get_factor_exposure(beta)", "米筐风险模型 beta 暴露"),
        ("betasq", "derived", "beta ** 2", "由 beta 平方得到"),
        ("bm", "direct", "rq.get_factor(book_to_market_ratio_lf)", "账面市值比 lf"),
        ("cash", "easy_construct", "cash_equivalent_mrq_0 / total_assets_mrq_0", "货币资金/总资产"),
        ("cashdebt", "direct", "rq.get_factor(ocf_to_debt_ttm)", "经营现金流/负债"),
        ("cfp", "direct", "rq.get_factor(cfp_ratio_ttm)", "现金收益率 ttm"),
        ("currat", "direct", "rq.get_factor(current_ratio_lf)", "流动比率 lf"),
        ("dy", "direct", "rq.get_factor(dividend_yield_ttm)", "股息率 ttm"),
        ("ep", "direct", "rq.get_factor(ep_ratio_ttm)", "盈市率 ttm"),
        ("lev", "direct", "rq.get_factor(book_leverage_lf)", "账面杠杆 lf"),
        ("mvel1", "direct", "rq.get_factor(market_cap_3)", "总市值"),
        ("quick", "direct", "rq.get_factor(quick_ratio_lf)", "速动比率 lf"),
        ("roaq", "direct", "rq.get_factor(return_on_asset_ttm)", "总资产报酬率 ttm"),
        ("roeq", "direct", "rq.get_factor(return_on_equity_ttm)", "净资产收益率 ttm"),
        ("sp", "direct", "rq.get_factor(sp_ratio_ttm)", "销售收益率 ttm"),
        ("turn", "direct", "rq.get_turnover_rate(...).month", "近一月平均换手率"),
        ("agr", "easy_construct", "total_assets_mrq_0 / total_assets_mrq_4 - 1", "资产同比增长"),
        ("chcsho", "easy_construct", "pct_change_12m(shares_total)", "总股本同比变动"),
        ("chinv", "easy_construct", "inv/assets - inv_lag/assets_lag", "存货占资产变化"),
        ("convind", "easy_construct", "convertible stock_code membership", "是否有存续可转债"),
        ("dolvol", "easy_construct", "log1p(mean daily total_turnover)", "月内日均成交额取对数"),
        ("gma", "easy_construct", "gross_profit_ttm_0 / total_assets_mrq_0", "毛利/总资产"),
        ("hire", "easy_construct", "pct_change_12m(staff_count_latest)", "员工数同比变化"),
        ("maxret", "easy_construct", "max daily close pct_change in month", "月内最大日收益"),
        ("mom1m", "easy_construct", "pct_change_1m(close_month_end)", "1月动量"),
        ("mom6m", "easy_construct", "pct_change_6m(close_month_end)", "6月动量"),
        ("mom12m", "easy_construct", "pct_change_12m(close_month_end)", "12月动量"),
        ("mom36m", "easy_construct", "pct_change_36m(close_month_end)", "36月动量"),
        ("chmom", "easy_construct", "mom6m - prior_6m_mom", "动量变化"),
        ("retvol", "easy_construct", "std daily close pct_change in month", "月内日收益波动率"),
        ("saleinv", "easy_construct", "operating_revenue_ttm_0 / inventory_mrq_0", "销售/存货"),
        ("salerec", "easy_construct", "operating_revenue_ttm_0 / net_accts_receivable_mrq_0", "销售/应收"),
        ("sgr", "easy_construct", "operating_revenue_ttm_0 / operating_revenue_ttm_4 - 1", "销售同比增长"),
        ("std_dolvol", "easy_construct", "rolling_12m_std(dolvol)", "成交额波动"),
        ("std_turn", "easy_construct", "rolling_12m_std(turn)", "换手率波动"),
        ("tang", "easy_construct", "Berger-style tangibility ratio", "有形资产占比"),
        ("zerotrade", "easy_construct", "share of zero-volume days in month", "零成交日占比"),
    ]
    return pd.DataFrame(records, columns=["gu_name", "method_type", "source_method", "note"])


def main() -> None:
    rq.init()
    instruments = build_universe_master()
    month_end_dates = get_month_end_dates(START_DATE, END_DATE)
    feature_catalog = build_feature_catalog()
    catalog_path = OUTPUT_DIR / "rqdata_direct_easy_feature_catalog.csv"
    feature_catalog.to_csv(catalog_path, index=False, encoding="utf-8-sig")

    progress_records: list[dict[str, object]] = []
    for idx, month_end in enumerate(month_end_dates, start=1):
        part_path = PARTS_DIR / f"{month_end:%Y%m}.parquet"
        if part_path.exists():
            existing_rows = len(pd.read_parquet(part_path, columns=["order_book_id"]))
            progress_records.append(
                {"date": month_end, "rows": existing_rows, "status": "skipped_existing"}
            )
            pd.DataFrame(progress_records).to_csv(PROGRESS_PATH, index=False, encoding="utf-8-sig")
            print(
                f"[{idx}/{len(month_end_dates)}] skipped existing {month_end.date()} rows={existing_rows}",
                flush=True,
            )
            continue

        ids = eligible_ids(instruments, month_end)
        month_start = month_end.replace(day=1)
        factor_df = fetch_factor_block(ids, month_end)
        risk_df = fetch_risk_block(ids, month_end)
        turnover_df = fetch_turnover_block(ids, month_end)
        shares_df = fetch_shares_block(ids, month_end)
        price_df = fetch_price_block(ids, month_start, month_end)
        price_stats = compute_monthly_price_stats(price_df, ids)
        staff_df = fetch_staff_block(ids, month_end)
        convertible_stock_set = fetch_convertible_stock_set(month_end)

        month_frame = compute_month_frame(
            ids=ids,
            month_end=month_end,
            factor_df=factor_df,
            risk_df=risk_df,
            turnover_df=turnover_df,
            shares_df=shares_df,
            price_stats=price_stats,
            staff_df=staff_df,
            convertible_stock_set=convertible_stock_set,
        )
        month_frame = month_frame.reset_index()
        month_frame.to_parquet(part_path, index=False)
        progress_records.append({"date": month_end, "rows": len(month_frame), "status": "built"})
        pd.DataFrame(progress_records).to_csv(PROGRESS_PATH, index=False, encoding="utf-8-sig")
        print(f"[{idx}/{len(month_end_dates)}] built {month_end.date()} rows={len(month_frame)}", flush=True)

    part_files = sorted(PARTS_DIR.glob("*.parquet"))
    raw_monthly = pd.concat((pd.read_parquet(path) for path in part_files), axis=0, ignore_index=True)
    final_panel = finalize_panel(raw_monthly)

    raw_path = OUTPUT_DIR / "rqdata_direct_easy_raw_monthly.parquet"
    panel_path = OUTPUT_DIR / "rqdata_direct_easy_characteristics_monthly.parquet"

    raw_monthly.to_parquet(raw_path, index=False)
    final_panel.to_parquet(panel_path, index=False)

    print(f"saved raw monthly panel to: {raw_path}", flush=True)
    print(f"saved final feature panel to: {panel_path}", flush=True)
    print(f"saved feature catalog to: {catalog_path}", flush=True)


if __name__ == "__main__":
    main()
