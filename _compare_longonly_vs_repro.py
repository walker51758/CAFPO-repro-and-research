"""Side-by-side comparison: long-only softmax vs original PPO+log-return (long-short)."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("rqdata_output")
LONGONLY_RET = OUT / "top200_softmax_longonly" / "cafpo_ppo_softmax_longonly_top200_returns.csv"
LONGONLY_SUM = OUT / "top200_softmax_longonly" / "cafpo_ppo_softmax_longonly_top200_summary.csv"
REPRO_RET = OUT / "cafpo_ppo_log_rolling_returns.csv"
REPRO_SUM = OUT / "cafpo_ppo_log_table1_summary.csv"

print("== files ==")
for p in [LONGONLY_RET, LONGONLY_SUM, REPRO_RET, REPRO_SUM]:
    print(f"  {p.name}: exists={p.exists()} rows={p.stat().st_size if p.exists() else 0} bytes")

lo_returns = pd.read_csv(LONGONLY_RET, parse_dates=["date"])
re_returns = pd.read_csv(REPRO_RET, parse_dates=["date"])
lo_summary = pd.read_csv(LONGONLY_SUM)
re_summary = pd.read_csv(REPRO_SUM)

print("\n== long-only softmax summary (already displayed) ==")
print(lo_summary.to_string(index=False))

print("\n== repro PPO long-short summary (already in rqdata_output) ==")
print(re_summary.to_string(index=False))

print("\n== long-only method name(s) ==")
print(lo_returns["method"].unique())
print("\n== repro method name(s) ==")
print(re_returns["method"].unique())

lo_returns["date"] = pd.to_datetime(lo_returns["date"])
re_returns["date"] = pd.to_datetime(re_returns["date"])

# Per-test-year breakdown
def per_year_table(df, label):
    out = []
    for ty, g in df.groupby("test_year"):
        r = g["portfolio_return"].dropna()
        if len(r) == 0:
            continue
        std = float(r.std(ddof=1))
        mean = float(r.mean())
        sr_m = mean / std if std > 0 else float("nan")
        wealth = (1.0 + r).cumprod()
        dd = float((wealth / wealth.cummax() - 1.0).min())
        cr = float(wealth.iloc[-1] - 1.0)
        out.append({"test_year": int(ty), "method_set": label, "n_months": int(len(r)),
                    "mean_monthly": mean, "std_monthly": std,
                    "sharpe_monthly": sr_m, "compound_return": cr, "max_drawdown": dd})
    return pd.DataFrame(out)


re_ppo = re_returns[re_returns["method"] == "CAFPO_PPO_LogReturn_5SeedAvg"]
lo_ppo = lo_returns[lo_returns["method"].str.contains("SoftmaxLongOnly")]

per_re = per_year_table(re_ppo, "repro_PPO_LS")
per_lo = per_year_table(lo_ppo, "longonly_PPO_SOFTMAX")

combined = pd.concat([per_re, per_lo], ignore_index=True).sort_values(["test_year", "method_set"])
print("\n== per-test-year PPO breakdown ==")
print(combined.round(5).to_string(index=False))

# Side-by-side per year diff
pivot = combined.pivot_table(index="test_year", columns="method_set", values=["sharpe_monthly", "compound_return", "max_drawdown", "mean_monthly"])
print("\n== pivot (year x metric x method_set) ==")
print(pivot.round(5).to_string())

# Aggregate
def aggregate(df, name):
    r = df["portfolio_return"].dropna()
    std = float(r.std(ddof=1))
    mean = float(r.mean())
    sr_a = mean / std * np.sqrt(12) if std > 0 else float("nan")
    wealth = (1.0 + r).cumprod()
    return {"method": name, "n_months": int(len(r)),
            "mean_monthly": mean, "std_monthly": std,
            "sharpe_annual": sr_a,
            "compound_return": float(wealth.iloc[-1] - 1.0),
            "max_drawdown": float((wealth / wealth.cummax() - 1.0).min())}

agg = pd.DataFrame([
    aggregate(re_ppo, "repro_PPO_LS (long-short)"),
    aggregate(lo_ppo, "longonly_PPO_SOFTMAX (long-only)"),
])
print("\n== aggregate comparison ==")
print(agg.round(5).to_string(index=False))

# Side-by-side baseline comparison
print("\n== baseline aggregate ==")
base_rows = []
for method in re_returns["method"].unique():
    if method == "CAFPO_PPO_LogReturn_5SeedAvg":
        continue
    base_rows.append(aggregate(re_returns[re_returns["method"] == method], f"REPRO::{method}"))
for method in lo_returns["method"].unique():
    if "SoftmaxLongOnly" in method:
        continue
    base_rows.append(aggregate(lo_returns[lo_returns["method"] == method], f"LONGONLY::{method}"))
print(pd.DataFrame(base_rows).round(5).to_string(index=False))

# Correlation across monthly returns
common_dates = sorted(set(re_ppo["date"]) & set(lo_ppo["date"]))
re_r = re_ppo.set_index("date")["portfolio_return"].reindex(common_dates)
lo_r = lo_ppo.set_index("date")["portfolio_return"].reindex(common_dates)
corr = float(np.corrcoef(re_r, lo_r)[0, 1])
print(f"\n== correlation across common test months (n={len(common_dates)}): {corr:+.3f} ==")
print(f"repro_PPO mean={re_r.mean():+.5f}  std={re_r.std(ddof=1):.5f}")
print(f"longonly mean={lo_r.mean():+.5f}  std={lo_r.std(ddof=1):.5f}")

# Sign agreement
print(f"\n== sign agreement (same sign share): {float((np.sign(re_r) == np.sign(lo_r)).mean()):.3f} ==")
print(f"repro > longonly share: {float((re_r > lo_r).mean()):.3f}")
print(f"avg absolute gap (monthly return): {float((re_r - lo_r).abs().mean()):.5f}")

# Plot wealth curves
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 5))
for method, group in re_returns.groupby("method"):
    g = group.sort_values("date")
    ax.plot(g["date"], (1.0 + g["portfolio_return"]).cumprod(), label=f"REPRO::{method}", linewidth=1, alpha=0.7)
for method, group in lo_returns.groupby("method"):
    g = group.sort_values("date")
    ax.plot(g["date"], (1.0 + g["portfolio_return"]).cumprod(), label=f"LONGONLY::{method}", linewidth=1, alpha=0.7)
ax.set_title("Repro PPO+LS vs Long-Only Softmax — out-of-sample wealth")
ax.set_ylabel("wealth")
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "compare_longonly_vs_repro_wealth.png", dpi=120)
print(f"\nSaved: {OUT / 'compare_longonly_vs_repro_wealth.png'}")

# Per-year wealth
fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=False)
for ax_, (ty, gr_re) in zip(axes.flatten(), re_ppo.groupby("test_year")):
    gr_re = gr_re.sort_values("date")
    gr_lo = lo_ppo[lo_ppo["test_year"] == ty].sort_values("date")
    if len(gr_re) == 0 or len(gr_lo) == 0:
        ax_.set_title(f"test_year={ty}: (skipped, missing data)")
        continue
    ax_.plot(gr_re["date"], (1.0 + gr_re["portfolio_return"]).cumprod(), marker="o", label="repro PPO LS", color="tab:blue")
    ax_.plot(gr_lo["date"], (1.0 + gr_lo["portfolio_return"]).cumprod(), marker="s", label="longonly softmax", color="tab:red")
    ax_.axhline(1.0, color="black", lw=0.5)
    ax_.set_title(f"test_year={ty}")
    ax_.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "compare_longonly_vs_repro_per_year_wealth.png", dpi=120)
print(f"Saved: {OUT / 'compare_longonly_vs_repro_per_year_wealth.png'}")
