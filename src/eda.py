from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ARTIFACTS_DIR, DATA_PATH, FIGURES_DIR, ID_COLUMN, TARGET

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "#2E6F9E"


def savefig(name):
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def run_eda(data_path: Path = DATA_PATH):
    df = pd.read_csv(data_path)
    if TARGET not in df:
        raise ValueError(f"No existe {TARGET}")
    numeric = [c for c in df.select_dtypes(include=np.number).columns if c not in [TARGET, ID_COLUMN]]
    categorical = list(df.select_dtypes(exclude=np.number).columns)

    summary = {
        "rows": int(df.shape[0]), "columns": int(df.shape[1]),
        "numeric_features": len(numeric), "categorical_features": len(categorical),
        "missing_cells": int(df.isna().sum().sum()),
        "columns_with_missing": int(df.isna().any().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df else None,
        "target": {
            "mean": float(df[TARGET].mean()), "median": float(df[TARGET].median()),
            "std": float(df[TARGET].std()), "min": float(df[TARGET].min()),
            "max": float(df[TARGET].max()), "skew": float(df[TARGET].skew()),
        },
        "numeric_coded_categories": ["MSSubClass", "MoSold"],
    }
    (ARTIFACTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    desc = df[numeric + [TARGET]].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    desc["median"] = df[numeric + [TARGET]].median()
    desc["range"] = desc["max"] - desc["min"]
    desc["skew"] = df[numeric + [TARGET]].skew(numeric_only=True)
    desc.to_csv(ARTIFACTS_DIR / "numeric_summary.csv")

    missing = pd.DataFrame({"missing_count": df.isna().sum(), "missing_pct": df.isna().mean() * 100})
    missing = missing[missing.missing_count > 0].sort_values("missing_pct", ascending=False)
    missing.to_csv(ARTIFACTS_DIR / "missing_summary.csv")
    pd.DataFrame({"dtype": df.dtypes.astype(str), "unique": df.nunique(dropna=True),
                  "missing": df.isna().sum()}).to_csv(ARTIFACTS_DIR / "feature_inventory.csv")

    corr = df.select_dtypes(include=np.number).corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    corr.rename("pearson_correlation").to_csv(ARTIFACTS_DIR / "target_correlations.csv")
    cat_effect = []
    for col in categorical:
        grouped = df.groupby(col, dropna=False)[TARGET].agg(["count", "mean", "median"])
        between = grouped["mean"].std()
        cat_effect.append({"feature": col, "unique": int(df[col].nunique(dropna=True)),
                           "between_group_mean_std": float(between) if pd.notna(between) else 0.0})
    pd.DataFrame(cat_effect).sort_values("between_group_mean_std", ascending=False).to_csv(
        ARTIFACTS_DIR / "categorical_target_effects.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df[TARGET], kde=True, ax=axes[0], color=PALETTE)
    axes[0].set_title("Distribucion de SalePrice (escala original)")
    sns.histplot(np.log1p(df[TARGET]), kde=True, ax=axes[1], color="#C7793B")
    axes[1].set_title("Distribucion de log1p(SalePrice)")
    axes[1].set_xlabel("log1p(SalePrice)")
    savefig("target_distribution.png")

    plt.figure(figsize=(10, 6))
    plot_missing = missing.sort_values("missing_pct")
    plt.barh(plot_missing.index, plot_missing.missing_pct, color=PALETTE)
    plt.xlabel("Porcentaje de valores faltantes")
    plt.title("Valores faltantes por variable")
    savefig("missing_values.png")

    top_features = corr.head(14).index.tolist() + [TARGET]
    plt.figure(figsize=(11, 9))
    sns.heatmap(df[top_features].corr(), cmap="vlag", center=0, annot=True, fmt=".2f", square=False)
    plt.title("Correlaciones numericas principales")
    savefig("correlation_heatmap.png")

    plt.figure(figsize=(9, 6))
    ordered = corr.head(15).sort_values()
    colors = ["#B84A3A" if v < 0 else PALETTE for v in ordered]
    plt.barh(ordered.index, ordered.values, color=colors)
    plt.xlabel("Correlacion de Pearson con SalePrice")
    plt.title("Variables numericas mas relacionadas con el precio")
    savefig("top_correlations.png")

    scatter = [c for c in ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt", "1stFlrSF"] if c in df]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, col in zip(axes.flat, scatter):
        sns.scatterplot(data=df, x=col, y=TARGET, alpha=.55, s=24, ax=ax, color=PALETTE)
        sns.regplot(data=df, x=col, y=TARGET, scatter=False, ax=ax, color="#B84A3A")
        ax.set_title(f"{col} vs. SalePrice")
    savefig("key_scatterplots.png")

    box_cols = [c for c in ["GrLivArea", "LotArea", "TotalBsmtSF", "GarageArea", TARGET] if c in df]
    fig, axes = plt.subplots(1, len(box_cols), figsize=(3.1 * len(box_cols), 5))
    for ax, col in zip(axes, box_cols):
        sns.boxplot(y=df[col], ax=ax, color="#8DB3C7")
        ax.set_title(col)
    savefig("outlier_boxplots.png")

    cat_cols = [c for c in ["Neighborhood", "ExterQual", "KitchenQual", "GarageFinish"] if c in df]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, col in zip(axes.flat, cat_cols):
        plot_df = df[[col, TARGET]].copy()
        plot_df[col] = plot_df[col].fillna("Missing").astype(str)
        order = plot_df.groupby(col)[TARGET].median().sort_values().index
        sns.boxplot(data=plot_df, x=col, y=TARGET, order=order, ax=ax, color="#B9D7C3", showfliers=False)
        ax.tick_params(axis="x", rotation=60)
        ax.set_title(f"Precio por {col}")
    plt.tight_layout()
    savefig("categorical_price_relationships.png")

    hist_cols = [c for c in ["GrLivArea", "LotArea", "TotalBsmtSF", "YearBuilt", "OverallQual", "GarageArea"] if c in df]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, hist_cols):
        sns.histplot(df[col], kde=True, ax=ax, color=PALETTE)
        ax.set_title(col)
    plt.tight_layout()
    savefig("numeric_distributions.png")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_eda()
