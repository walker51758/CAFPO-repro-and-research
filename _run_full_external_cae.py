"""
Full rolling CAFPO/PPO run using external CAE factors.
Runs the same logic as `run_main_ppo_log_experiment` in the notebook,
but as a standalone script. This will be started in background (async).

Usage:
    conda run -n drl_env python _run_full_external_cae.py

Note: This will run the full rolling experiment and may take a long time
on CPU. Outputs are written under the module's `OUTPUT_DIR / 'reproduction_external'`.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cafpo_reproduction import (
    CafpoPortfolioEnv,
    build_rolling_splits,
    load_model_panel,
    make_ppo_model,
    panel_to_arrays,
    train_cae,
    extract_cae_outputs,
    normalize_long_short,
    set_seed,
    evaluate_sb3_ensemble,
    run_historical_sort_baseline,
    run_markowitz_baseline,
    performance_summary,
    save_cae_outputs,
    OUTPUT_DIR,
)

# Experiment knobs (match notebook defaults)
N_FACTORS = 5
LOOKBACK = 12
CAE_EPOCHS = 200
CAE_PATIENCE = 20
PPO_TOTAL_TIMESTEPS = 20_000
PPO_N_STEPS = 64
PPO_BATCH_SIZE = 32
DRL_SEEDS = [101, 202, 303, 404, 505]
TOP_BOTTOM_FRAC = 0.30

EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "reproduction_external"
EXPERIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_full():
    set_seed(42)
    print("[full] loading model panel...")
    df, feature_cols = load_model_panel()
    arrays = panel_to_arrays(df, feature_cols, max_stocks=200)
    splits = build_rolling_splits(arrays.dates, train_years=10, test_years=1)

    all_returns = []
    all_histories = []

    for s in splits:
        print(f"[full] running test_year={s.test_year} train={s.train_start_year}-{s.train_end_year}")
        val_idx = s.train_idx[-12:]
        fit_idx = s.train_idx[:-12]
        try:
            cae_model, cae_history = train_cae(
                arrays,
                train_idx=fit_idx,
                val_idx=val_idx,
                n_factors=N_FACTORS,
                epochs=CAE_EPOCHS,
                patience=CAE_PATIENCE,
                seed=42 + int(s.test_year),
                cae_mode="external",
            )
        except Exception as exc:
            print(f"[full] train_cae failed for test_year={s.test_year}: {exc}")
            continue
        if "test_year" not in cae_history.columns:
            cae_history.insert(0, "test_year", s.test_year)
        all_histories.append(cae_history)

        factors, betas, reconstructed = extract_cae_outputs(cae_model, arrays)
        save_cae_outputs(factors, betas, reconstructed, arrays, EXPERIMENT_OUTPUT_DIR / f"cafpo_cae_outputs_test_{s.test_year}")

        obs, _ = CafpoPortfolioEnv(
            factors=factors,
            forward_returns=arrays.ret_forward,
            mask=arrays.mask,
            dates=arrays.dates,
            indices=s.train_idx[LOOKBACK:],
            lookback=LOOKBACK,
            reward_mode="log_return",
        ).reset()

        # Baselines
        equal_weight = run_historical_sort_baseline(
            "EqualWeight", arrays, train_idx=s.train_idx, test_idx=s.test_idx, top_frac=TOP_BOTTOM_FRAC, value_weighted=False
        )
        value_weight = run_historical_sort_baseline(
            "ValueWeight", arrays, train_idx=s.train_idx, test_idx=s.test_idx, top_frac=TOP_BOTTOM_FRAC, value_weighted=True
        )
        markowitz = run_markowitz_baseline(arrays, train_idx=s.train_idx, test_idx=s.test_idx)
        for frame in (equal_weight, value_weight, markowitz):
            frame.insert(0, "experiment", "reproduction_external")
            frame.insert(1, "test_year", s.test_year)
        all_returns.extend([equal_weight, value_weight, markowitz])

        # PPO ensemble
        models = []
        for seed in DRL_SEEDS:
            train_env = CafpoPortfolioEnv(
                factors=factors,
                forward_returns=arrays.ret_forward,
                mask=arrays.mask,
                dates=arrays.dates,
                indices=s.train_idx[LOOKBACK:],
                lookback=LOOKBACK,
                reward_mode="log_return",
            )
            model = make_ppo_model(train_env, seed=seed + int(s.test_year), n_steps=PPO_N_STEPS, batch_size=PPO_BATCH_SIZE, verbose=0)
            model.learn(total_timesteps=PPO_TOTAL_TIMESTEPS, progress_bar=False)
            models.append(model)
            print(f"[full] seed={seed} done for test_year={s.test_year}")

        test_env = CafpoPortfolioEnv(
            factors=factors,
            forward_returns=arrays.ret_forward,
            mask=arrays.mask,
            dates=arrays.dates,
            indices=s.test_idx,
            lookback=LOOKBACK,
            reward_mode="log_return",
        )
        ppo = evaluate_sb3_ensemble(models, test_env)
        ppo.insert(0, "experiment", "reproduction_external")
        ppo.insert(1, "method", "CAFPO_PPO_LogReturn_5SeedAvg")
        ppo.insert(2, "test_year", s.test_year)
        all_returns.append(ppo)

        # persist partials
        partial = pd.concat(all_returns, ignore_index=True) if all_returns else pd.DataFrame()
        partial.to_csv(EXPERIMENT_OUTPUT_DIR / f"partial_returns_test_{s.test_year}.csv", index=False, encoding="utf-8-sig")
        print(f"[full] saved partial returns for test_year={s.test_year} rows={len(partial)}")

    # final aggregation and summary
    returns = pd.concat(all_returns, ignore_index=True) if all_returns else pd.DataFrame()
    history = pd.concat(all_histories, ignore_index=True) if all_histories else pd.DataFrame()
    summary = performance_summary(returns)

    returns.to_csv(EXPERIMENT_OUTPUT_DIR / "cafpo_ppo_log_rolling_returns.csv", index=False, encoding="utf-8-sig")
    history.to_csv(EXPERIMENT_OUTPUT_DIR / "cafpo_cae_rolling_training_history.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(EXPERIMENT_OUTPUT_DIR / "cafpo_ppo_log_table1_summary.csv", index=False, encoding="utf-8-sig")
    print("[full] DONE")


if __name__ == "__main__":
    run_full()
