#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistical significance testing across models for a chosen metric.
- Normality per model (Shapiro–Wilk)
- Homogeneity of variances (Levene)
- Main test: One-way ANOVA (if all normal) else Kruskal–Wallis
- Post-hoc: Games–Howell (normal) OR Dunn (non-normal, Bonferroni)
- Boxplot + heatmap of pairwise p-values
Usage:
  python stats_by_model.py --input results/asr_evaluation.json --metric wer_mean
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import shapiro, levene, f_oneway, kruskal
import pingouin as pg
import scikit_posthocs as sp


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="results/asr_evaluation.json",
                    help="Path to ASR evaluation JSON")
    ap.add_argument("--metric", type=str, default="wer_mean",
                    help="Metric column to analyze (e.g., wer_mean, cer_mean, latency_mean, rtf_mean)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    return ap.parse_args()


# ---------------------------
# Helpers
# ---------------------------
def p_to_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def safe_shapiro(x: pd.Series):
    """Run Shapiro–Wilk but handle constant or small samples gracefully."""
    x = pd.Series(x).dropna()
    try:
        if len(x) < 3 or x.nunique() <= 1:
            return np.nan, np.nan, "not_applicable"
        stat, p = shapiro(x)
        return stat, p, "ok"
    except Exception:
        return np.nan, np.nan, "error"

def ensure_dirs():
    Path("figs").mkdir(exist_ok=True, parents=True)
    Path("tables").mkdir(exist_ok=True, parents=True)

def heatmap_from_square_pmat(pmat: pd.DataFrame, title: str, outpath: Path):
    m = pmat.copy().astype(float)
    np.fill_diagonal(m.values, np.nan)
    annot = m.applymap(lambda p: p_to_stars(p) if pd.notnull(p) else "")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        m, annot=annot, fmt="", vmin=0, vmax=1, cmap="viridis",
        cbar_kws={"label": "p-value"}, linewidths=.5, linecolor='white'
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def heatmap_from_gameshowell_long(gh_df: pd.DataFrame, title: str, outpath: Path):
    groups = sorted(set(gh_df['A']).union(set(gh_df['B'])))
    pmat = pd.DataFrame(np.nan, index=groups, columns=groups, dtype=float)
    for _, r in gh_df.iterrows():
        a, b, p = r['A'], r['B'], float(r['pval'])
        pmat.loc[a, b] = p
        pmat.loc[b, a] = p
    heatmap_from_square_pmat(pmat, title, outpath)


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    INPUT_JSON = args.input
    METRIC_COL = args.metric
    ALPHA = args.alpha

    ensure_dirs()

    # -------- Load & reshape to long DF --------
    with open(INPUT_JSON) as f:
        results = json.load(f)

    records = []
    for model_name, model_data in results.items():
        for dialect, metrics in model_data["per_dialect"].items():
            row = {
                "model": model_name,
                "dialect": dialect,
                # Available metric fields – extend if your JSON contains means for others
                "wer_mean": metrics["wer"]["mean"],
                "wer_std":  metrics["wer"]["std"],
                
            }
            # If you also store CER/latency/rtf means in your JSON, add them to row here:
            # row["cer_mean"] = metrics["cer"]["mean"]
            row["latency_mean"] = metrics["latency"]["mean"]
            # row["rtf_mean"] = metrics["rtf"]["mean"]
            records.append(row)

    df = pd.DataFrame(records)

    if METRIC_COL not in df.columns:
        raise ValueError(f"Metric '{METRIC_COL}' not found in DataFrame columns: {list(df.columns)}")

    # -------- Boxplot across models --------
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="model", y=METRIC_COL, data=df)
    plt.title(f"{METRIC_COL} distribution across models")
    plt.tight_layout()
    plt.savefig("figs/boxplot_models.png", dpi=200)
    plt.close()

    # -------- Group values by model --------
    models = df["model"].unique()
    groups = [df[df["model"] == m][METRIC_COL].dropna() for m in models]

    # -------- Normality per model --------
    print("\n=== Normality per model (Shapiro–Wilk) ===")
    normal_flags = []
    for m in models:
        vals = df[df["model"] == m][METRIC_COL]
        stat, p, status = safe_shapiro(vals)
        if status == "not_applicable":
            print(f"{m}: Shapiro–Wilk not applicable (constant or too small sample)")
            normal_flags.append(False)  # conservative
        elif status == "error":
            print(f"{m}: Shapiro–Wilk error; treating as non-normal")
            normal_flags.append(False)
        else:
            print(f"{m}: W={stat:.3f}, p={p:.4f}")
            normal_flags.append(p >= ALPHA)
    is_all_normal = all(normal_flags)

    # -------- Homogeneity (Levene) --------
    stat, p = levene(*groups)
    print("\n=== Homogeneity of variances (Levene) ===")
    print(f"Levene: stat={stat:.3f}, p={p:.4f}")

    # -------- Main test + Post-hoc --------
    if is_all_normal:
        stat, p = f_oneway(*groups)
        print("\n=== One-way ANOVA ===")
        print(f"F={stat:.3f}, p={p:.4f}")

        if p < ALPHA:
            print("→ Significant overall effect; running Games–Howell post-hoc ...")
            gh = pg.pairwise_gameshowell(dv=METRIC_COL, between="model", data=df)
            gh.to_csv("tables/posthoc_gameshowell_models.csv", index=False)
            print("Saved: tables/posthoc_gameshowell_models.csv")
            heatmap_from_gameshowell_long(
                gh,
                title=f"Games–Howell post-hoc ({METRIC_COL}) — pairwise p-values (models)",
                outpath=Path("figs/posthoc_gameshowell_models.png"),
            )
            print("Saved: figs/posthoc_gameshowell_models.png")
        else:
            print("→ No significant differences; no post-hoc needed.")
    else:
        stat, p = kruskal(*groups)
        print("\n=== Kruskal–Wallis ===")
        print(f"H={stat:.3f}, p={p:.4f}")

        if p < ALPHA:
            print("→ Significant overall effect; running Dunn (Bonferroni) post-hoc ...")
            dunn = sp.posthoc_dunn(df, val_col=METRIC_COL, group_col='model', p_adjust='bonferroni')
            dunn.to_csv("tables/posthoc_dunn_models.csv")
            print("Saved: tables/posthoc_dunn_models.csv")
            heatmap_from_square_pmat(
                dunn,
                title=f"Dunn post-hoc ({METRIC_COL}) — pairwise p-values (models)",
                outpath=Path("figs/posthoc_dunn_models.png"),
            )
            print("Saved: figs/posthoc_dunn_models.png")
        else:
            print("→ No significant differences; no post-hoc needed.")

    # -------- Summary --------
    print("\n=== Summary (models) ===")
    print(f"Metric: {METRIC_COL}")
    print(f"All groups normal? {'Yes' if is_all_normal else 'No'}")
    if is_all_normal:
        print("Main test: One-way ANOVA (Games–Howell post-hoc if significant).")
    else:
        print("Main test: Kruskal–Wallis (Dunn post-hoc with Bonferroni if significant).")
    print("Outputs:")
    print(" - figs/boxplot_models.png")
    print(" - figs/posthoc_gameshowell_models.png or figs/posthoc_dunn_models.png (if significant)")
    print(" - tables/posthoc_gameshowell_models.csv or tables/posthoc_dunn_models.csv (if significant)")


if __name__ == "__main__":
    main()
