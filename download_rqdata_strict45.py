from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rqdatac as rq


BATCH_SIZE = 1200
MIN_IDIO_OBS = 10

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "rqdata_output"
OUTPUT_DIR.mkdir(exist_ok=True)
PARTS_DIR = OUTPUT_DIR / "strict45_monthly_parts"
PARTS_DIR.mkdir(exist_ok=True)
PROGRESS_PATH = OUTPUT_DIR / "strict45_progress.csv"

BASE_RAW_PATH = OUTPUT_DIR / "rqdata_direct_easy_raw_monthly.parquet"
BASE_CHAR_PATH = OUTPUT_DIR / "rqdata_direct_easy_characteristics_monthly.parquet"
STRICT45_RAW_PATH = OUTPUT_DIR / "rqdata_strict45_raw_monthly.parquet"
STRICT45_CHAR_PATH = OUTPUT_DIR / "rqdata_strict45_characteristics_monthly.parquet"
COMBINED82_PATH = OUTPUT_DIR / "rqdata_82_characteristics_monthly.parquet"
CATALOG_PATH = OUTPUT_DIR / "rqdata_strict45_feature_catalog.csv"

EXTRA_FACTOR_FIELDS = [
    "equity_parent_company_mrq_0",
    "equity_parent_company_mrq_4",
    "cash_flow_from_operating_activities_ttm_0",
    "cash_flow_from_operating_activities_ttm_4",
    "net_profit_parent_company_ttm_0",
    "net_profit_parent_company_ttm_4",
    "total_liabilities_mrq_0",
    "total_liabilities_mrq_4",
    "current_assets_mrq_0",
    "current_assets_mrq_4",
    "current_liabilities_mrq_0",
    "current_liabilities_mrq_4",
    "cash_equivalent_mrq_4",
    "net_accts_receivable_mrq_4",
    "total_fixed_assets_mrq_4",
    "gross_profit_ttm_4",
    "gross_profit_margin_ttm",
    "net_profit_margin_ttm",
    "long_term_loans_mrq_0",
    "long_term_loans_mrq_4",
    "bond_payable_mrq_0",
    "bond_payable_mrq_4",
    "long_term_payable_mrq_0",
    "long_term_payable_mrq_4",
    "depreciation_and_amortization_ttm_0",
    "depreciation_and_amortization_ttm_4",
    "income_tax_ttm_0",
    "income_tax_ttm_4",
    "profit_before_tax_ttm_0",
    "profit_before_tax_ttm_4",
    "selling_expense_ttm_0",
    "selling_expense_ttm_4",
    "ga_expense_ttm_0",
    "ga_expense_ttm_4",
    "real_estate_investment_mrq_0",
    "operating_profitTTM",
    "return_on_invested_capital_ttm",
    "total_asset_turnover_ttm",
    "net_asset_growth_ratio_lf",
    "short_term_loans_mrq_0",
    "short_term_loans_mrq_4",
    "tax_payable_mrq_0",
    "tax_payable_mrq_4",
]

STRICT45_FEATURES = [
    "absacc",
    "acc",
    "age",
    "bm ia",
    "cashpr",
    "cfp ia",
    "chatoia",
    "chempia",
    "chpmia",
    "chtx",
    "cinvest",
    "depr",
    "divi",
    "divo",
    "egr",
    "grCAPX",
    "grltnoa",
    "herf",
    "idiovol",
    "ill",
    "indmom",
    "invest",
    "lgr",
    "ms",
    "mve ia",
    "nincr",
    "operprof",
    "pchcapx ia",
    "pchcurrat",
    "pchdepr",
    "pchgm pchsale",
    "pchquick",
    "pchsale pchinvt",
    "pchsale pchrect",
    "pchsale pchxsga",
    "pchsaleinv",
    "pctacc",
    "ps",
    "realestate",
    "roavol",
    "roic",
    "salecash",
    "stdacc",
    "stdcf",
    "tb",
]


def batch_list(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


def safe_div(numer: pd.Series | np.ndarray, denom: pd.Series | np.ndarray) -> pd.Series:
    out = pd.Series(numer) / pd.Series(denom)
    return out.replace([np.inf, -np.inf], np.nan)


def safe_div_frame(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = numer / denom
    return out.replace([np.inf, -np.inf], np.nan)


def build_universe_master() -> pd.DataFrame:
    instruments = rq.all_instruments("CS").copy()
    instruments["listed_date"] = pd.to_datetime(instruments["listed_date"], errors="coerce")
    return instruments[["order_book_id", "listed_date"]]


def fetch_factor_block(ids: list[str], date: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in batch_list(ids, BATCH_SIZE):
        df = rq.get_factor(chunk, EXTRA_FACTOR_FIELDS, start_date=date, end_date=date)
        if isinstance(df.index, pd.MultiIndex):
            if "date" in df.index.names:
                df = df.xs(date, level="date")
            else:
                df.index = df.index.get_level_values(0)
        df.index.name = "order_book_id"
        frames.append(df)
    factor_df = pd.concat(frames, axis=0) if frames else pd.DataFrame(index=pd.Index([], name="order_book_id"))
    return factor_df.reindex(ids)


def fetch_industry_block(ids: list[str], date: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in batch_list(ids, BATCH_SIZE):
        df = rq.get_instrument_industry(chunk, source="citics_2019", level=1, date=date)
        if df is None or df.empty:
            tmp = pd.DataFrame(index=pd.Index(chunk, name="order_book_id"))
        else:
            tmp = df.copy()
            tmp.index.name = "order_book_id"
        frames.append(tmp)
    industry_df = pd.concat(frames, axis=0) if frames else pd.DataFrame(index=pd.Index([], name="order_book_id"))
    return industry_df.reindex(ids)


def fetch_price_block(ids: list[str], month_start: pd.Timestamp, month_end: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in batch_list(ids, BATCH_SIZE):
        df = rq.get_price(
            chunk,
            start_date=month_start,
            end_date=month_end,
            frequency="1d",
            fields=["close", "total_turnover"],
        )
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["close", "total_turnover"])
    return pd.concat(frames, axis=0)


def compute_monthly_microstructure(price_df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=pd.Index(ids, name="order_book_id"))
    out["ill"] = np.nan
    out["idiovol"] = np.nan
    if price_df.empty:
        return out

    daily = price_df.reset_index().sort_values(["order_book_id", "date"]).copy()
    daily["daily_ret"] = daily.groupby("order_book_id")["close"].pct_change()
    market_ret = daily.groupby("date")["daily_ret"].mean().rename("market_ret")
    daily = daily.merge(market_ret, on="date", how="left")
    daily["ill_component"] = (daily["daily_ret"].abs() / daily["total_turnover"]).replace(
        [np.inf, -np.inf], np.nan
    )

    ill = daily.groupby("order_book_id")["ill_component"].mean()

    def calc_idiovol(group: pd.DataFrame) -> float:
        sub = group[["daily_ret", "market_ret"]].dropna()
        if len(sub) < MIN_IDIO_OBS:
            return np.nan
        x = sub["market_ret"].to_numpy(dtype=float)
        y = sub["daily_ret"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        if len(resid) < 2:
            return np.nan
        return float(np.std(resid, ddof=1))

    idiovol = daily.groupby("order_book_id", sort=False).apply(calc_idiovol)
    out["ill"] = ill.reindex(ids)
    out["idiovol"] = idiovol.reindex(ids)
    return out


def build_month_supplement(
    ids: list[str],
    date: pd.Timestamp,
    listed_lookup: pd.Series,
    factor_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    micro_df: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=pd.Index(ids, name="order_book_id"))
    frame["date"] = date
    frame["listed_date"] = listed_lookup.reindex(ids)
    frame = frame.join(industry_df)
    frame = frame.join(factor_df)
    frame = frame.join(micro_df)
    frame = frame.rename(
        columns={
            "first_industry_code": "industry_code",
            "first_industry_name": "industry_name",
        }
    )
    return frame.reset_index()


def rowwise_std(series_list: list[pd.Series]) -> pd.Series:
    stacked = pd.concat(series_list, axis=1)
    return stacked.std(axis=1, ddof=1)


def binary_signal(condition: pd.Series, valid: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=condition.index, dtype=float)
    out.loc[valid] = condition.loc[valid].astype(float)
    return out


def finalize_strict45(base_raw: pd.DataFrame, supplement_raw: pd.DataFrame) -> pd.DataFrame:
    panel = base_raw.merge(supplement_raw, on=["date", "order_book_id"], how="left")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["listed_date"] = pd.to_datetime(panel["listed_date"])
    panel = panel.sort_values(["order_book_id", "date"]).copy()
    grouped = panel.groupby("order_book_id", group_keys=False)
    industry_group = panel.groupby(["date", "industry_code"], group_keys=False)

    ltd_0 = (
        panel["long_term_loans_mrq_0"].fillna(0)
        + panel["bond_payable_mrq_0"].fillna(0)
        + panel["long_term_payable_mrq_0"].fillna(0)
    )
    ltd_4 = (
        panel["long_term_loans_mrq_4"].fillna(0)
        + panel["bond_payable_mrq_4"].fillna(0)
        + panel["long_term_payable_mrq_4"].fillna(0)
    )
    wc_0 = (
        panel["current_assets_mrq_0"]
        - panel["cash_equivalent_mrq_0"]
        - panel["current_liabilities_mrq_0"]
        + panel["short_term_loans_mrq_0"].fillna(0)
        + panel["tax_payable_mrq_0"].fillna(0)
    )
    wc_4 = (
        panel["current_assets_mrq_4"]
        - panel["cash_equivalent_mrq_4"]
        - panel["current_liabilities_mrq_4"]
        + panel["short_term_loans_mrq_4"].fillna(0)
        + panel["tax_payable_mrq_4"].fillna(0)
    )
    avg_assets = (panel["total_assets_mrq_0"] + panel["total_assets_mrq_4"]) / 2.0
    acc_numer = (wc_0 - wc_4) - panel["depreciation_and_amortization_ttm_0"]
    cfo_assets = safe_div_frame(
        panel["cash_flow_from_operating_activities_ttm_0"], panel["total_assets_mrq_0"]
    )
    current_ratio_0 = safe_div_frame(panel["current_assets_mrq_0"], panel["current_liabilities_mrq_0"])
    current_ratio_4 = safe_div_frame(panel["current_assets_mrq_4"], panel["current_liabilities_mrq_4"])
    quick_ratio_0 = safe_div_frame(
        panel["current_assets_mrq_0"] - panel["inventory_mrq_0"], panel["current_liabilities_mrq_0"]
    )
    quick_ratio_4 = safe_div_frame(
        panel["current_assets_mrq_4"] - panel["inventory_mrq_4"], panel["current_liabilities_mrq_4"]
    )
    depr_ratio = safe_div_frame(
        panel["depreciation_and_amortization_ttm_0"], panel["total_fixed_assets_mrq_0"]
    )
    net_profit_margin_lag12 = grouped["net_profit_margin_ttm"].shift(12)
    gross_profit_margin_lag12 = grouped["gross_profit_margin_ttm"].shift(12)
    asset_turnover_lag12 = grouped["total_asset_turnover_ttm"].shift(12)
    saleinv_lag12 = grouped["saleinv"].shift(12)
    dy_lag12 = grouped["dy"].shift(12)

    capx = (
        panel["total_fixed_assets_mrq_0"]
        - panel["total_fixed_assets_mrq_4"]
        + panel["depreciation_and_amortization_ttm_0"].fillna(0)
    ).clip(lower=0)
    capx_lag12 = grouped["total_fixed_assets_mrq_0"].shift(12) - grouped["total_fixed_assets_mrq_4"].shift(12)
    capx_lag12 = (capx_lag12 + grouped["depreciation_and_amortization_ttm_0"].shift(12).fillna(0)).clip(lower=0)
    capx_intensity = safe_div_frame(capx, panel["total_assets_mrq_4"])

    sales_growth_q = grouped["operating_revenue_ttm_0"].pct_change(3)

    lt_oper_assets_0 = panel["total_assets_mrq_0"] - panel["current_assets_mrq_0"] - panel["cash_equivalent_mrq_0"]
    lt_oper_assets_4 = panel["total_assets_mrq_4"] - panel["current_assets_mrq_4"] - panel["cash_equivalent_mrq_4"]
    lt_oper_liab_0 = panel["total_liabilities_mrq_0"] - panel["current_liabilities_mrq_0"] - ltd_0
    lt_oper_liab_4 = panel["total_liabilities_mrq_4"] - panel["current_liabilities_mrq_4"] - ltd_4
    ltnoa_ratio_0 = safe_div_frame(lt_oper_assets_0 - lt_oper_liab_0, panel["total_assets_mrq_0"])
    ltnoa_ratio_4 = safe_div_frame(lt_oper_assets_4 - lt_oper_liab_4, panel["total_assets_mrq_4"])

    panel["acc"] = safe_div_frame(acc_numer, avg_assets)
    panel["absacc"] = panel["acc"].abs()
    panel["age"] = (panel["date"] - panel["listed_date"]).dt.days / 365.25
    panel["bm ia"] = panel["bm"] - industry_group["bm"].transform("mean")
    panel["cashpr"] = safe_div_frame(
        panel["mvel1"] + panel["total_liabilities_mrq_0"] - panel["total_assets_mrq_0"],
        panel["cash_equivalent_mrq_0"],
    )
    panel["cfp ia"] = panel["cfp"] - industry_group["cfp"].transform("mean")
    asset_turnover_delta = panel["total_asset_turnover_ttm"] - asset_turnover_lag12
    panel["chatoia"] = asset_turnover_delta - asset_turnover_delta.groupby(
        [panel["date"], panel["industry_code"]]
    ).transform("mean")
    panel["chempia"] = panel["hire"] - industry_group["hire"].transform("mean")
    npm_delta = panel["net_profit_margin_ttm"] - net_profit_margin_lag12
    panel["chpmia"] = npm_delta - npm_delta.groupby([panel["date"], panel["industry_code"]]).transform("mean")
    panel["chtx"] = safe_div_frame(panel["income_tax_ttm_0"], panel["total_assets_mrq_0"]) - safe_div_frame(
        panel["income_tax_ttm_4"], panel["total_assets_mrq_4"]
    )
    panel["cinvest"] = capx_intensity
    panel["depr"] = depr_ratio
    panel["divi"] = ((panel["dy"] > 0) & ((dy_lag12 <= 0) | dy_lag12.isna())).astype(float)
    panel["divo"] = ((panel["dy"] <= 0) & (dy_lag12 > 0)).astype(float)
    panel["egr"] = safe_div_frame(panel["equity_parent_company_mrq_0"], panel["equity_parent_company_mrq_4"]) - 1.0
    panel["grCAPX"] = safe_div_frame(capx, capx_lag12) - 1.0
    panel["grltnoa"] = ltnoa_ratio_0 - ltnoa_ratio_4

    industry_sales = industry_group["operating_revenue_ttm_0"].transform("sum")
    sales_share = safe_div_frame(panel["operating_revenue_ttm_0"], industry_sales)
    panel["herf"] = (sales_share**2).groupby([panel["date"], panel["industry_code"]]).transform("sum")
    panel["indmom"] = industry_group["mom6m"].transform("mean")
    panel["invest"] = safe_div_frame(
        (panel["total_fixed_assets_mrq_0"] - panel["total_fixed_assets_mrq_4"])
        + (panel["inventory_mrq_0"] - panel["inventory_mrq_4"]),
        panel["total_assets_mrq_4"],
    )
    panel["lgr"] = safe_div_frame(ltd_0, ltd_4) - 1.0
    panel["mve ia"] = np.log(panel["mvel1"]).replace([np.inf, -np.inf], np.nan) - industry_group["mvel1"].transform(
        lambda s: np.log(s).replace([np.inf, -np.inf], np.nan).mean()
    )
    ni_0 = panel["net_profit_parent_company_ttm_0"]
    ni_3 = grouped["net_profit_parent_company_ttm_0"].shift(3)
    ni_6 = grouped["net_profit_parent_company_ttm_0"].shift(6)
    ni_9 = grouped["net_profit_parent_company_ttm_0"].shift(9)
    ni_12 = grouped["net_profit_parent_company_ttm_0"].shift(12)
    panel["nincr"] = pd.concat(
        [
            binary_signal(ni_0 > ni_3, ni_0.notna() & ni_3.notna()),
            binary_signal(ni_3 > ni_6, ni_3.notna() & ni_6.notna()),
            binary_signal(ni_6 > ni_9, ni_6.notna() & ni_9.notna()),
            binary_signal(ni_9 > ni_12, ni_9.notna() & ni_12.notna()),
        ],
        axis=1,
    ).sum(axis=1, min_count=4)
    panel["operprof"] = safe_div_frame(panel["operating_profitTTM"], panel["equity_parent_company_mrq_0"])
    panel["pchcurrat"] = safe_div_frame(current_ratio_0, current_ratio_4) - 1.0
    panel["pchgm pchsale"] = safe_div_frame(panel["gross_profit_margin_ttm"], gross_profit_margin_lag12) - 1.0 - panel["sgr"]
    panel["pchquick"] = safe_div_frame(quick_ratio_0, quick_ratio_4) - 1.0
    panel["pchsale pchinvt"] = panel["sgr"] - (safe_div_frame(panel["inventory_mrq_0"], panel["inventory_mrq_4"]) - 1.0)
    panel["pchsale pchrect"] = panel["sgr"] - (
        safe_div_frame(panel["net_accts_receivable_mrq_0"], panel["net_accts_receivable_mrq_4"]) - 1.0
    )
    xsga_0 = panel["selling_expense_ttm_0"].fillna(0) + panel["ga_expense_ttm_0"].fillna(0)
    xsga_4 = panel["selling_expense_ttm_4"].fillna(0) + panel["ga_expense_ttm_4"].fillna(0)
    panel["pchsale pchxsga"] = panel["sgr"] - (safe_div_frame(xsga_0, xsga_4) - 1.0)
    panel["pchsaleinv"] = safe_div_frame(panel["saleinv"], saleinv_lag12) - 1.0
    panel["pctacc"] = safe_div_frame(acc_numer, panel["net_profit_parent_company_ttm_0"].abs())
    panel["realestate"] = safe_div_frame(panel["real_estate_investment_mrq_0"], panel["total_assets_mrq_0"])
    panel["roic"] = panel["return_on_invested_capital_ttm"]
    panel["salecash"] = safe_div_frame(panel["operating_revenue_ttm_0"], panel["cash_equivalent_mrq_0"])
    panel["tb"] = safe_div_frame(panel["income_tax_ttm_0"], panel["profit_before_tax_ttm_0"])

    roaq_q0 = panel["roaq"]
    roaq_q3 = grouped["roaq"].shift(3)
    roaq_q6 = grouped["roaq"].shift(6)
    roaq_q9 = grouped["roaq"].shift(9)
    panel["roavol"] = rowwise_std([roaq_q0, roaq_q3, roaq_q6, roaq_q9])

    acc_q0 = panel["acc"]
    acc_q3 = grouped["acc"].shift(3) if "acc" in panel.columns else acc_q0.groupby(panel["order_book_id"]).shift(3)
    acc_q6 = acc_q0.groupby(panel["order_book_id"]).shift(6)
    acc_q9 = acc_q0.groupby(panel["order_book_id"]).shift(9)
    panel["stdacc"] = rowwise_std([acc_q0, acc_q3, acc_q6, acc_q9])

    cfo_q0 = cfo_assets
    cfo_q3 = cfo_assets.groupby(panel["order_book_id"]).shift(3)
    cfo_q6 = cfo_assets.groupby(panel["order_book_id"]).shift(6)
    cfo_q9 = cfo_assets.groupby(panel["order_book_id"]).shift(9)
    panel["stdcf"] = rowwise_std([cfo_q0, cfo_q3, cfo_q6, cfo_q9])

    sales_growth_q3 = sales_growth_q.groupby(panel["order_book_id"]).shift(3)
    sales_growth_q6 = sales_growth_q.groupby(panel["order_book_id"]).shift(6)
    sales_growth_q9 = sales_growth_q.groupby(panel["order_book_id"]).shift(9)
    sales_growth_var = rowwise_std([sales_growth_q, sales_growth_q3, sales_growth_q6, sales_growth_q9])
    sga_intensity = safe_div_frame(xsga_0, panel["total_assets_mrq_0"])

    lever_0 = safe_div_frame(ltd_0, panel["total_assets_mrq_0"])
    lever_4 = safe_div_frame(ltd_4, panel["total_assets_mrq_4"])
    shares_growth_12 = grouped["shares_total"].pct_change(12)
    panel["ps"] = pd.concat(
        [
            binary_signal(panel["roaq"] > 0, panel["roaq"].notna()),
            binary_signal(
                panel["cash_flow_from_operating_activities_ttm_0"] > 0,
                panel["cash_flow_from_operating_activities_ttm_0"].notna(),
            ),
            binary_signal(panel["roaq"] > grouped["roaq"].shift(12), panel["roaq"].notna() & grouped["roaq"].shift(12).notna()),
            binary_signal(cfo_assets > panel["roaq"], cfo_assets.notna() & panel["roaq"].notna()),
            binary_signal(lever_0 < lever_4, lever_0.notna() & lever_4.notna()),
            binary_signal(current_ratio_0 > current_ratio_4, current_ratio_0.notna() & current_ratio_4.notna()),
            binary_signal(shares_growth_12 <= 0, shares_growth_12.notna()),
            binary_signal(
                panel["gross_profit_margin_ttm"] > gross_profit_margin_lag12,
                panel["gross_profit_margin_ttm"].notna() & gross_profit_margin_lag12.notna(),
            ),
            binary_signal(
                panel["total_asset_turnover_ttm"] > asset_turnover_lag12,
                panel["total_asset_turnover_ttm"].notna() & asset_turnover_lag12.notna(),
            ),
        ],
        axis=1,
    ).sum(axis=1, min_count=9)

    roa_ind_median = industry_group["roaq"].transform("median")
    cfo_ind_median = cfo_assets.groupby([panel["date"], panel["industry_code"]]).transform("median")
    roavol_ind_median = panel.groupby(["date", "industry_code"])["roavol"].transform("median")
    sales_var_ind_median = sales_growth_var.groupby([panel["date"], panel["industry_code"]]).transform("median")
    capx_ind_median = capx_intensity.groupby([panel["date"], panel["industry_code"]]).transform("median")
    sga_ind_median = sga_intensity.groupby([panel["date"], panel["industry_code"]]).transform("median")
    gpm_ind_median = industry_group["gross_profit_margin_ttm"].transform("median")

    panel["ms"] = pd.concat(
        [
            binary_signal(panel["roaq"] > roa_ind_median, panel["roaq"].notna() & roa_ind_median.notna()),
            binary_signal(cfo_assets > cfo_ind_median, cfo_assets.notna() & cfo_ind_median.notna()),
            binary_signal(cfo_assets > panel["roaq"], cfo_assets.notna() & panel["roaq"].notna()),
            binary_signal(panel["roavol"] < roavol_ind_median, panel["roavol"].notna() & roavol_ind_median.notna()),
            binary_signal(sales_growth_var < sales_var_ind_median, sales_growth_var.notna() & sales_var_ind_median.notna()),
            binary_signal(capx_intensity > capx_ind_median, capx_intensity.notna() & capx_ind_median.notna()),
            binary_signal(sga_intensity > sga_ind_median, sga_intensity.notna() & sga_ind_median.notna()),
            binary_signal(
                panel["gross_profit_margin_ttm"] > gpm_ind_median,
                panel["gross_profit_margin_ttm"].notna() & gpm_ind_median.notna(),
            ),
        ],
        axis=1,
    ).sum(axis=1, min_count=8)

    panel["pchcapx ia"] = panel["grCAPX"] - industry_group["grCAPX"].transform("mean")
    panel["pchdepr"] = safe_div_frame(depr_ratio, depr_ratio.groupby(panel["order_book_id"]).shift(12)) - 1.0

    out_cols = ["date", "order_book_id"] + STRICT45_FEATURES
    return panel[out_cols].copy()


def build_feature_catalog() -> pd.DataFrame:
    records = [
        ("absacc", "proxy_construct", "abs(acc)", "基于 Sloan 工作资本应计近似"),
        ("acc", "proxy_construct", "working capital accruals / avg assets", "使用 current assets/liabilities、cash、short-term debt、tax payable、depr 近似"),
        ("age", "a_share_adapted", "(date - listed_date) / 365.25", "A股用上市年限替代 Compustat 覆盖年限"),
        ("bm ia", "easy_construct", "bm - industry mean(bm)", "中信2019一级行业"),
        ("cashpr", "proxy_construct", "(mvel1 + liabilities - assets) / cash", "现金生产率近似"),
        ("cfp ia", "easy_construct", "cfp - industry mean(cfp)", "中信2019一级行业"),
        ("chatoia", "proxy_construct", "delta asset turnover - industry mean delta", "asset turnover 用 ttm 因子并按月 shift 12"),
        ("chempia", "easy_construct", "hire - industry mean(hire)", "中信2019一级行业"),
        ("chpmia", "proxy_construct", "delta profit margin - industry mean delta", "profit margin 用 ttm 因子并按月 shift 12"),
        ("chtx", "easy_construct", "income_tax/assets - lag income_tax/assets", "同比口径"),
        ("cinvest", "proxy_construct", "capx / lag assets", "capx 用 fixed assets change + depreciation 近似"),
        ("depr", "easy_construct", "depreciation / fixed assets", "折旧摊销 / 固定资产"),
        ("divi", "easy_construct", "dy>0 and lag12 dy<=0", "用 ttm 股息率近似分红启动"),
        ("divo", "easy_construct", "dy<=0 and lag12 dy>0", "用 ttm 股息率近似分红停止"),
        ("egr", "easy_construct", "equity_0 / equity_4 - 1", "归母权益同比增长"),
        ("grCAPX", "proxy_construct", "capx / lag12 capx - 1", "capx 用 fixed assets change + depreciation 近似"),
        ("grltnoa", "proxy_construct", "ltnoa_ratio_0 - ltnoa_ratio_4", "长期净经营资产同比变化"),
        ("herf", "easy_construct", "industry sum((sales / industry_sales)^2)", "行业销售集中度"),
        ("idiovol", "proxy_construct", "std(residual) from daily market-model", "月内日收益对等权市场收益回归残差波动"),
        ("ill", "easy_construct", "mean(abs(daily_ret) / total_turnover)", "Amihud 月度非流动性"),
        ("indmom", "easy_construct", "industry mean(mom6m)", "行业动量"),
        ("invest", "proxy_construct", "(delta fixed assets + delta inventory) / lag assets", "资本开支与存货联合投资近似"),
        ("lgr", "easy_construct", "ltd_0 / ltd_4 - 1", "长期债务同比增长"),
        ("ms", "proxy_construct", "8-signal growth-score proxy", "Mohanram 分数使用 A 股可得变量代理"),
        ("mve ia", "proxy_construct", "log(mvel1) - industry mean(log(mvel1))", "行业调整规模"),
        ("nincr", "proxy_construct", "count of positive 3m-spaced earnings increases", "按月面板近似季度连增"),
        ("operprof", "easy_construct", "operating_profitTTM / equity", "经营盈利能力"),
        ("pchcapx ia", "proxy_construct", "grCAPX - industry mean(grCAPX)", "行业调整资本开支变化"),
        ("pchcurrat", "easy_construct", "current_ratio_0 / current_ratio_4 - 1", "同比变化"),
        ("pchdepr", "proxy_construct", "depr / lag12 depr - 1", "按月面板近似年度同比"),
        ("pchgm pchsale", "proxy_construct", "gross margin growth - sgr", "gross margin 用 ttm 因子并按月 shift 12"),
        ("pchquick", "easy_construct", "quick_ratio_0 / quick_ratio_4 - 1", "同比变化"),
        ("pchsale pchinvt", "easy_construct", "sgr - inventory growth", "销售增速减存货增速"),
        ("pchsale pchrect", "easy_construct", "sgr - receivable growth", "销售增速减应收增速"),
        ("pchsale pchxsga", "easy_construct", "sgr - xsga growth", "销售增速减销管费增速"),
        ("pchsaleinv", "easy_construct", "saleinv / lag12 saleinv - 1", "销售存货比同比变化"),
        ("pctacc", "proxy_construct", "accruals / abs(net income)", "基于 Sloan 工作资本应计近似"),
        ("ps", "proxy_construct", "9-signal Piotroski-style score", "F-score 用 A 股可得字段构造"),
        ("realestate", "easy_construct", "real_estate_investment / assets", "投资性房地产占比"),
        ("roavol", "proxy_construct", "std of roaq at 0/3/6/9m shifts", "按月面板近似季度波动"),
        ("roic", "direct", "return_on_invested_capital_ttm", "米筐直接因子"),
        ("salecash", "easy_construct", "sales / cash", "销售对现金"),
        ("stdacc", "proxy_construct", "std of acc at 0/3/6/9m shifts", "按月面板近似季度波动"),
        ("stdcf", "proxy_construct", "std of cfo/assets at 0/3/6/9m shifts", "按月面板近似季度波动"),
        ("tb", "easy_construct", "income_tax / profit_before_tax", "税账差近似"),
    ]
    return pd.DataFrame(records, columns=["gu_name", "method_type", "source_method", "note"])


def main() -> None:
    rq.init()
    base_raw = pd.read_parquet(BASE_RAW_PATH)
    base_chars = pd.read_parquet(BASE_CHAR_PATH)
    base_raw["date"] = pd.to_datetime(base_raw["date"])
    base_chars["date"] = pd.to_datetime(base_chars["date"])
    extra_base_cols = [col for col in base_chars.columns if col not in {"date", "order_book_id"} and col not in base_raw.columns]
    base_panel = base_raw.merge(base_chars[["date", "order_book_id"] + extra_base_cols], on=["date", "order_book_id"], how="left")
    listed_df = build_universe_master()
    listed_lookup = listed_df.set_index("order_book_id")["listed_date"]
    feature_catalog = build_feature_catalog()
    feature_catalog.to_csv(CATALOG_PATH, index=False, encoding="utf-8-sig")

    month_end_dates = sorted(base_raw["date"].drop_duplicates().tolist())
    month_limit = int(os.getenv("RQDATA_MONTH_LIMIT", "0") or "0")
    if month_limit > 0:
        month_end_dates = month_end_dates[:month_limit]

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

        ids = base_raw.loc[base_raw["date"] == month_end, "order_book_id"].tolist()
        month_start = month_end.replace(day=1)
        factor_df = fetch_factor_block(ids, month_end)
        industry_df = fetch_industry_block(ids, month_end)
        price_df = fetch_price_block(ids, month_start, month_end)
        micro_df = compute_monthly_microstructure(price_df, ids)
        month_frame = build_month_supplement(
            ids=ids,
            date=month_end,
            listed_lookup=listed_lookup,
            factor_df=factor_df,
            industry_df=industry_df,
            micro_df=micro_df,
        )
        month_frame.to_parquet(part_path, index=False)
        progress_records.append({"date": month_end, "rows": len(month_frame), "status": "built"})
        pd.DataFrame(progress_records).to_csv(PROGRESS_PATH, index=False, encoding="utf-8-sig")
        print(f"[{idx}/{len(month_end_dates)}] built {month_end.date()} rows={len(month_frame)}", flush=True)

    part_files = sorted(PARTS_DIR.glob("*.parquet"))
    supplement_raw = pd.concat((pd.read_parquet(path) for path in part_files), axis=0, ignore_index=True)
    strict45 = finalize_strict45(base_panel, supplement_raw)
    combined82 = base_chars.merge(strict45, on=["date", "order_book_id"], how="left")

    supplement_raw.to_parquet(STRICT45_RAW_PATH, index=False)
    strict45.to_parquet(STRICT45_CHAR_PATH, index=False)
    combined82.to_parquet(COMBINED82_PATH, index=False)

    print(f"saved supplement raw panel to: {STRICT45_RAW_PATH}", flush=True)
    print(f"saved strict45 feature panel to: {STRICT45_CHAR_PATH}", flush=True)
    print(f"saved combined 82 feature panel to: {COMBINED82_PATH}", flush=True)
    print(f"saved feature catalog to: {CATALOG_PATH}", flush=True)


if __name__ == "__main__":
    main()
