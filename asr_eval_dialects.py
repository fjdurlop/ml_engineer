#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistical significance testing across dialects for WER.
- Normality per dialect (Shapiro–Wilk)
- Homogeneity of variances (Levene)
- Welch's ANOVA (if normal) OR Kruskal–Wallis (if non-normal)
- Post-hoc: Games–Howell (normal) OR Dunn (non-normal, Bonferroni)
- Boxplot + heatmap of pairwise p-values
"""

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import shapiro, levene, f_oneway, kruskal

# Post-hoc libs
import pingouin as pg
import scikit_posthocs as sp


# ---------------------------
# Config
# ---------------------------
INPUT_JSON = "results/asr_evaluation.json"
ALPHA = 0.05
METRIC_COL = "wer_mean"   # change to "cer_mean", "latency_mean", etc. if you extend your JSON


# ---------------------------
# Helpers
# ---------------------------
def p_to_stars(p: float) -> str:
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def heatmap_from_dunn(pmat: pd.DataFrame, title: str, outpath: Path):
    m = pmat.copy().astype(float)
    np.fill_diagonal(m.values, np.nan)
    annot = m.applymap(lambda p: p_to_stars(p) if pd.notnull(p) else "")
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        m, annot=annot, fmt="", vmin=0, vmax=1, cmap="viridis",
        cbar_kws={"label": "p-value"}, linewidths=.5, linecolor='white'
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def heatmap_from_gameshowell(gh_df: pd.DataFrame, title: str, outpath: Path):
    groups = sorted(set(gh_df['A']).union(set(gh_df['B'])))
    pmat = pd.DataFrame(np.nan, index=groups, columns=groups, dtype=float)
    for _, r in gh_df.iterrows():
        a, b, p = r['A'], r['B'], float(r['pval'])
        pmat.loc[a, b] = p
        pmat.loc[b, a] = p
    heatmap_from_dunn(pmat, title, outpath)  # same renderer works


def safe_shapiro(x: pd.Series):
    """Run Shapiro–Wilk but handle constant or small samples gracefully."""
    x = pd.Series(x).dropna()
    try:
        # Shapiro requires n in [3, 5000] and some variance
        if len(x) < 3 or x.nunique() <= 1:
            return np.nan, np.nan, "not_applicable"
        stat, p = shapiro(x)
        return stat, p, "ok"
    except Exception:
        return np.nan, np.nan, "error"


# ---------------------------
# Load & reshape
# ---------------------------
with open(INPUT_JSON) as f:
    results = json.load(f)

records = []
for model_name, model_data in results.items():
    for dialect, metrics in model_data["per_dialect"].items():
        records.append({
            "model": model_name,
            "dialect": dialect,
            "wer_mean": metrics["wer"]["mean"],
            "wer_std": metrics["wer"]["std"]
        })

df = pd.DataFrame(records)

# ---------------------------
# Boxplot across dialects
# ---------------------------
plt.figure(figsize=(7, 4))
sns.boxplot(x="dialect", y=METRIC_COL, data=df)
plt.title(f"{METRIC_COL} distribution across dialects")
plt.tight_layout()
Path("figs").mkdir(exist_ok=True, parents=True)
plt.savefig("figs/boxplot_dialects.png", dpi=200)
plt.close()

# ---------------------------
# Group values by dialect
# ---------------------------
dialects = df["dialect"].unique()
groups = [df[df["dialect"] == d][METRIC_COL].dropna() for d in dialects]

# ---------------------------
# Normality per dialect
# ---------------------------
print("\n=== Normality per dialect (Shapiro–Wilk) ===")
normal_flags = []
for d in dialects:
    vals = df[df["dialect"] == d][METRIC_COL]
    stat, p, status = safe_shapiro(vals)
    if status == "not_applicable":
        print(f"{d}: Shapiro–Wilk not applicable (constant or too small sample)")
        normal_flags.append(False)  # be conservative
    elif status == "error":
        print(f"{d}: Shapiro–Wilk error; treating as non-normal")
        normal_flags.append(False)
    else:
        print(f"{d}: W={stat:.3f}, p={p:.4f}")
        normal_flags.append(p >= ALPHA)

is_all_normal = all(normal_flags)

# ---------------------------
# Homogeneity (Levene)
# ---------------------------
stat, p = levene(*groups)
print("\n=== Homogeneity of variances (Levene) ===")
print(f"Levene: stat={stat:.3f}, p={p:.4f}")

# ---------------------------
# Main test + Post-hoc
# ---------------------------
Path("tables").mkdir(exist_ok=True, parents=True)
Path("figs").mkdir(exist_ok=True, parents=True)

if is_all_normal:
    # ANOVA (one-way). If you truly need Welch’s ANOVA (unequal variances),
    # you can use pingouin.welch_anova(dv=..., between=..., data=df_long)
    stat, p = f_oneway(*groups)
    print("\n=== One-way ANOVA ===")
    print(f"F={stat:.3f}, p={p:.4f}")

    if p < ALPHA:
        print("→ Significant overall effect; running Games–Howell post-hoc ...")
        gh = pg.pairwise_gameshowell(dv=METRIC_COL, between="dialect", data=df)
        gh.to_csv("tables/posthoc_gameshowell_dialects.csv", index=False)
        print("Saved: tables/posthoc_gameshowell_dialects.csv")
        # Heatmap
        heatmap_from_gameshowell(
            gh,
            title=f"Games–Howell post-hoc ({METRIC_COL}) — pairwise p-values",
            outpath=Path("figs/posthoc_gameshowell_dialects.png"),
        )
        print("Saved: figs/posthoc_gameshowell_dialects.png")
    else:
        print("→ No significant differences; no post-hoc needed.")
else:
    # Kruskal–Wallis
    stat, p = kruskal(*groups)
    print("\n=== Kruskal–Wallis ===")
    print(f"H={stat:.3f}, p={p:.4f}")

    if p < ALPHA:
        print("→ Significant overall effect; running Dunn (Bonferroni) post-hoc ...")
        dunn = sp.posthoc_dunn(df, val_col=METRIC_COL, group_col='dialect', p_adjust='bonferroni')
        dunn.to_csv("tables/posthoc_dunn_dialects.csv")
        print("Saved: tables/posthoc_dunn_dialects.csv")
        # Heatmap
        heatmap_from_dunn(
            dunn,
            title=f"Dunn post-hoc ({METRIC_COL}) — pairwise p-values",
            outpath=Path("figs/posthoc_dunn_dialects.png"),
        )
        print("Saved: figs/posthoc_dunn_dialects.png")
    else:
        print("→ No significant differences; no post-hoc needed.")

# ---------------------------
# Automated, human-readable summary
# ---------------------------
print("\n=== Summary (dialects) ===")
print(f"Metric: {METRIC_COL}")
print(f"All groups normal? {'Yes' if is_all_normal else 'No'}")
if is_all_normal:
    print("Main test: One-way ANOVA (Games–Howell post-hoc if significant).")
else:
    print("Main test: Kruskal–Wallis (Dunn post-hoc with Bonferroni if significant).")

print("Outputs:")
print(" - figs/boxplot_dialects.png")
print(" - figs/posthoc_gameshowell_dialects.png or figs/posthoc_dunn_dialects.png (if significant)")
print(" - tables/posthoc_gameshowell_dialects.csv or tables/posthoc_dunn_dialects.csv (if significant)")
