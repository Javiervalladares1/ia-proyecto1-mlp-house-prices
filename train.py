from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (ARTIFACTS_DIR, DATA_PATH, EXPERIMENTS_DIR, FIGURES_DIR,
                        ID_COLUMN, MODELS_DIR, PREDICTIONS_DIR, SEED, TARGET)
from src.model import (MLPConfig, predict_scaled, save_checkpoint, set_seed,
                       train_mlp, train_mlp_fixed)
from src.preprocessing import TargetTransformer, build_preprocessor


def rmse(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))


def price_bins(y, n=10):
    return pd.qcut(pd.Series(y).rank(method="first"), q=n, labels=False).to_numpy()


def jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def config_from_dict(params, seed=SEED):
    layers = params.get("hidden_layers", (256, 128, 64))
    if isinstance(layers, str):
        layers = tuple(int(x) for x in layers.split("-"))
    return MLPConfig(
        hidden_layers=tuple(layers), activation=params.get("activation", "relu"),
        dropout=float(params.get("dropout", 0.1)), normalization=params.get("normalization", "none"),
        learning_rate=float(params.get("learning_rate", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-4)), batch_size=int(params.get("batch_size", 64)),
        optimizer=params.get("optimizer", "adamw"), max_epochs=int(params.get("max_epochs", 350)),
        patience=int(params.get("patience", 35)), scheduler_patience=int(params.get("scheduler_patience", 10)),
        gradient_clip=float(params.get("gradient_clip", 5.0)), seed=seed,
    )


def prep_params(params):
    return {
        "scaler": params.get("scaler", "standard"),
        "skew_threshold": params.get("skew_threshold", 0.75),
        "clip_quantile": params.get("clip_quantile"),
        "min_frequency": params.get("min_frequency"),
        "feature_engineering": params.get("feature_engineering", True),
    }


def evaluate_config(X, y, params, folds=3, seed=SEED, trial=None, save_history_name=None):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_metrics, representative_history = [], None
    bins = price_bins(y, n=10)
    for fold, (tr, va) in enumerate(splitter.split(X, bins)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr].to_numpy(), y.iloc[va].to_numpy()
        preprocessor = build_preprocessor(Xtr, **prep_params(params))
        Xtr_p = preprocessor.fit_transform(Xtr).astype(np.float32)
        Xva_p = preprocessor.transform(Xva).astype(np.float32)
        target = TargetTransformer(params.get("target_transform", "raw")).fit(ytr)
        cfg = config_from_dict(params, seed=seed + fold)
        model, history, best_val = train_mlp(
            Xtr_p, target.transform(ytr).astype(np.float32), Xva_p, yva, target, cfg)
        best_row = min(history, key=lambda row: row["val_rmse"])
        fold_metrics.append({
            "fold": fold, "train_rmse": best_row["train_rmse"],
            "val_rmse": best_val, "best_epoch": best_row["epoch"],
            "n_features": Xtr_p.shape[1],
        })
        if fold == 0:
            representative_history = pd.DataFrame(history)
        if trial is not None:
            trial.report(float(np.mean([m["val_rmse"] for m in fold_metrics])), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if save_history_name and representative_history is not None:
        representative_history.to_csv(EXPERIMENTS_DIR / f"history_{save_history_name}.csv", index=False)
        plot_history(representative_history, FIGURES_DIR / f"training_{save_history_name}.png", save_history_name)
    return {
        "train_rmse": float(np.mean([m["train_rmse"] for m in fold_metrics])),
        "cv_rmse_mean": float(np.mean([m["val_rmse"] for m in fold_metrics])),
        "cv_rmse_std": float(np.std([m["val_rmse"] for m in fold_metrics], ddof=1)),
        "best_epoch_median": int(np.median([m["best_epoch"] for m in fold_metrics])),
        "n_features": int(np.median([m["n_features"] for m in fold_metrics])),
        "fold_metrics": fold_metrics,
    }


def plot_history(history, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(history.epoch, history.train_loss, label="Entrenamiento", color="#2E6F9E")
    axes[0].plot(history.epoch, history.val_loss, label="Validación", color="#C7793B")
    axes[0].set(title=f"Loss por época - {title}", xlabel="Época", ylabel="MSE estandarizado")
    axes[0].legend()
    axes[1].plot(history.epoch, history.train_rmse, label="Entrenamiento", color="#2E6F9E")
    axes[1].plot(history.epoch, history.val_rmse, label="Validación", color="#C7793B")
    axes[1].set(title=f"RMSE en escala original - {title}", xlabel="Época", ylabel="RMSE (USD)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fixed_history(history, path, title):
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(history.epoch, history.train_loss, color="#2E6F9E")
    axis.set(title=title, xlabel="Época", ylabel="MSE estandarizado")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def record_result(rows, experiment_id, kind, params, metrics, notes=""):
    cfg = config_from_dict(params)
    rows.append({
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(), "kind": kind, "seed": SEED,
        "features": "engineered" if params.get("feature_engineering", True) else "original",
        "preprocessing": json.dumps(prep_params(params), default=jsonable, sort_keys=True),
        "target_transform": params.get("target_transform", "raw"),
        "architecture": "-".join(map(str, cfg.hidden_layers)), "activation": cfg.activation,
        "normalization": cfg.normalization, "optimizer": cfg.optimizer,
        "learning_rate": cfg.learning_rate, "weight_decay": cfg.weight_decay,
        "dropout": cfg.dropout, "batch_size": cfg.batch_size,
        "epochs": metrics.get("best_epoch_median"), "early_stopping": kind != "final",
        "train_rmse": metrics.get("train_rmse"), "validation_rmse": metrics.get("cv_rmse_mean"),
        "cv_rmse_mean": metrics.get("cv_rmse_mean"), "cv_rmse_std": metrics.get("cv_rmse_std"),
        "holdout_rmse": metrics.get("holdout_rmse"), "n_features": metrics.get("n_features"),
        "notes": notes,
    })


def save_artifact(model_dir, model, preprocessor, target, cfg, prep, expected_columns, metadata):
    model_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, cfg, metadata["input_dim"], model_dir / "model.pt")
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
    joblib.dump(target, model_dir / "target_transformer.joblib")
    payload = {**metadata, "model_config": {**asdict(cfg), "hidden_layers": list(cfg.hidden_layers)},
               "preprocessing": prep, "expected_columns": list(expected_columns)}
    (model_dir / "metadata.json").write_text(json.dumps(payload, indent=2, default=jsonable), encoding="utf-8")


def fit_and_save(X, y, params, epochs, model_dir, metadata):
    prep = prep_params(params)
    preprocessor = build_preprocessor(X, **prep)
    Xp = preprocessor.fit_transform(X).astype(np.float32)
    target = TargetTransformer(params.get("target_transform", "raw")).fit(y)
    cfg = config_from_dict(params, seed=SEED)
    model, history = train_mlp_fixed(Xp, target.transform(y).astype(np.float32), cfg, epochs)
    metadata = {**metadata, "input_dim": int(Xp.shape[1]), "training_rows": int(len(X)), "fixed_epochs": int(epochs)}
    save_artifact(model_dir, model, preprocessor, target, cfg, prep, X.columns, metadata)
    return model, preprocessor, target, history


def error_analysis(X_hold, y_hold, pred, source_df):
    residual = y_hold - pred
    errors = pd.DataFrame({
        ID_COLUMN: source_df.loc[X_hold.index, ID_COLUMN].to_numpy() if ID_COLUMN in source_df else X_hold.index,
        "actual": y_hold, "prediction": pred, "residual": residual,
        "absolute_error": np.abs(residual), "absolute_percentage_error": np.abs(residual) / y_hold * 100,
    }, index=X_hold.index)
    errors = errors.join(source_df.loc[X_hold.index, [c for c in ["Neighborhood", "OverallQual", "GrLivArea", "YearBuilt"] if c in source_df]])
    sorted_errors = errors.sort_values("absolute_error", ascending=False)
    sorted_errors.to_csv(ARTIFACTS_DIR / "holdout_errors.csv", index=False)
    sorted_errors.head(20).to_csv(ARTIFACTS_DIR / "largest_errors.csv", index=False)
    errors["price_segment"] = pd.qcut(errors.actual, 4, labels=["Bajo", "Medio-bajo", "Medio-alto", "Alto"])
    segment = errors.groupby("price_segment", observed=True).agg(
        count=("actual", "size"), rmse=("residual", lambda x: np.sqrt(np.mean(x ** 2))),
        mae=("absolute_error", "mean"), bias=("residual", "mean")).reset_index()
    segment.to_csv(ARTIFACTS_DIR / "error_by_price_segment.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(y_hold, pred, alpha=.65, s=28, color="#2E6F9E")
    bounds = [min(y_hold.min(), pred.min()), max(y_hold.max(), pred.max())]
    axes[0, 0].plot(bounds, bounds, "--", color="#B84A3A")
    axes[0, 0].set(xlabel="Real", ylabel="Prediccion", title="Real vs. predicho")
    axes[0, 1].hist(residual, bins=28, color="#8DB3C7", edgecolor="white")
    axes[0, 1].axvline(0, color="#B84A3A", linestyle="--")
    axes[0, 1].set(title="Distribucion de residuos", xlabel="Real - predicho")
    axes[1, 0].scatter(pred, residual, alpha=.65, s=28, color="#2E6F9E")
    axes[1, 0].axhline(0, color="#B84A3A", linestyle="--")
    axes[1, 0].set(xlabel="Prediccion", ylabel="Residuo", title="Residuos vs. prediccion")
    axes[1, 1].bar(segment.price_segment.astype(str), segment.rmse, color="#C7793B")
    axes[1, 1].set(title="RMSE por rango de precio", ylabel="RMSE (USD)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_error_analysis.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return errors, segment


def main(trials=18):
    set_seed(SEED)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    dev_idx, hold_idx = train_test_split(np.arange(len(df)), test_size=.20, random_state=20260817,
                                         stratify=price_bins(y, 10))
    X_dev, y_dev = X.iloc[dev_idx].reset_index(drop=True), y.iloc[dev_idx].reset_index(drop=True)
    X_hold, y_hold = X.iloc[hold_idx], y.iloc[hold_idx].to_numpy()
    results = []

    exploratory = [
        ("baseline_raw", dict(hidden_layers="128-64", activation="relu", dropout=0.0,
          normalization="none", learning_rate=1e-3, weight_decay=1e-5, batch_size=64,
          optimizer="adam", target_transform="raw", scaler="standard", skew_threshold=None,
          clip_quantile=None, min_frequency=None, feature_engineering=False, max_epochs=300, patience=30)),
        ("baseline_log", dict(hidden_layers="128-64", activation="relu", dropout=0.0,
          normalization="none", learning_rate=1e-3, weight_decay=1e-5, batch_size=64,
          optimizer="adam", target_transform="log1p", scaler="standard", skew_threshold=.75,
          clip_quantile=None, min_frequency=None, feature_engineering=False, max_epochs=300, patience=30)),
        ("feature_engineered", dict(hidden_layers="256-128-64", activation="gelu", dropout=.10,
          normalization="layer", learning_rate=8e-4, weight_decay=1e-4, batch_size=64,
          optimizer="adamw", target_transform="raw", scaler="standard", skew_threshold=.75,
          clip_quantile=.005, min_frequency=2, feature_engineering=True, max_epochs=350, patience=35)),
        ("deep_low_regularization", dict(hidden_layers="512-256-128-64", activation="relu", dropout=0.0,
          normalization="none", learning_rate=1e-3, weight_decay=1e-7, batch_size=32,
          optimizer="adam", target_transform="raw", scaler="standard", skew_threshold=.75,
          clip_quantile=None, min_frequency=None, feature_engineering=True, max_epochs=350, patience=35)),
        ("robust_regularized", dict(hidden_layers="256-256-128", activation="silu", dropout=.2,
          normalization="layer", learning_rate=6e-4, weight_decay=5e-4, batch_size=64,
          optimizer="adamw", target_transform="raw", scaler="robust", skew_threshold=.75,
          clip_quantile=.01, min_frequency=2, feature_engineering=True, max_epochs=350, patience=35)),
    ]
    for name, params in exploratory:
        print(f"\n[exploratory] {name}", flush=True)
        metrics = evaluate_config(X_dev, y_dev, params, folds=3, save_history_name=name)
        record_result(results, name, "exploratory", params, metrics)
        pd.DataFrame(results).to_csv(EXPERIMENTS_DIR / "results.csv", index=False)
        print(metrics)

    def objective(trial):
        params = {
            "hidden_layers": trial.suggest_categorical("hidden_layers", ["64", "128-64", "256-128-64", "512-256-128", "256-256-128", "512-256-128-64"]),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "leaky_relu", "silu"]),
            "dropout": trial.suggest_float("dropout", 0.0, .35),
            "normalization": trial.suggest_categorical("normalization", ["none", "layer", "batch"]),
            "learning_rate": trial.suggest_float("learning_rate", 2e-4, 4e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "optimizer": trial.suggest_categorical("optimizer", ["adamw", "adam"]),
            "target_transform": trial.suggest_categorical("target_transform", ["raw", "log1p"]),
            "scaler": trial.suggest_categorical("scaler", ["standard", "robust"]),
            "skew_threshold": trial.suggest_categorical("skew_threshold", [None, .5, .75, 1.0]),
            "clip_quantile": trial.suggest_categorical("clip_quantile", [None, .005, .01]),
            "min_frequency": trial.suggest_categorical("min_frequency", [None, 2, 5]),
            "feature_engineering": trial.suggest_categorical("feature_engineering", [True, False]),
            "max_epochs": 350, "patience": 35,
        }
        metrics = evaluate_config(X_dev, y_dev, params, folds=3, trial=trial)
        trial.set_user_attr("metrics", metrics)
        return metrics["cv_rmse_mean"]

    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                study_name="ames_mlp", storage=f"sqlite:///{EXPERIMENTS_DIR / 'optuna.db'}",
                                load_if_exists=True)
    existing_trials = len(study.trials)
    if existing_trials < 2:
        study.enqueue_trial({
            "hidden_layers": "256-256-128", "activation": "silu", "dropout": .2,
            "normalization": "layer", "learning_rate": 6e-4, "weight_decay": 5e-4,
            "batch_size": 64, "optimizer": "adamw", "target_transform": "raw",
            "scaler": "robust", "skew_threshold": .75, "clip_quantile": .01,
            "min_frequency": 2, "feature_engineering": True,
        })
    remaining = max(0, trials - existing_trials)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True, show_progress_bar=False)
    study.trials_dataframe().to_csv(EXPERIMENTS_DIR / "optuna_trials.csv", index=False)
    for completed_trial in study.trials:
        if completed_trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        metrics = completed_trial.user_attrs.get("metrics")
        if metrics is None:
            raise RuntimeError(f"El trial completo {completed_trial.number} no contiene métricas reproducibles")
        params = {**completed_trial.params, "max_epochs": 350, "patience": 35}
        record_result(results, f"optuna_{completed_trial.number:03d}", "optuna", params, metrics)

    # Re-check three promising, structurally different candidates on a new
    # five-fold partition. This reduces the chance of selecting a lucky 3-fold trial.
    candidates = [
        ("optuna_winner", dict(study.best_trial.params)),
        ("robust_regularized", dict(exploratory[4][1])),
        ("feature_engineered", dict(exploratory[2][1])),
    ]
    confirmations = []
    for candidate_name, candidate_params in candidates:
        candidate_params.update(max_epochs=450, patience=50)
        print(f"\n[confirmatory 5-fold] {candidate_name}", candidate_params, flush=True)
        candidate_metrics = evaluate_config(X_dev, y_dev, candidate_params, folds=5, seed=31415,
                                            save_history_name=f"confirm_{candidate_name}")
        confirmations.append((candidate_metrics["cv_rmse_mean"], candidate_name,
                              candidate_params, candidate_metrics))
        record_result(results, f"confirm_{candidate_name}", "confirmatory", candidate_params,
                      candidate_metrics, notes="Compared before examining the internal holdout")
    _, selected_name, best_params, confirm = min(confirmations, key=lambda item: item[0])
    plot_history(pd.read_csv(EXPERIMENTS_DIR / f"history_confirm_{selected_name}.csv"),
                 FIGURES_DIR / "training_final_cv.png", "final_cv")

    # The epoch count is selected only from confirmatory development CV.
    final_epochs = max(20, int(confirm["best_epoch_median"]))

    # Honest internal holdout: fit for the fixed CV-derived epoch count. The
    # holdout is used once for reporting, never for checkpoint/model selection.
    pre = build_preprocessor(X_dev, **prep_params(best_params))
    Xd = pre.fit_transform(X_dev).astype(np.float32)
    Xh = pre.transform(X_hold).astype(np.float32)
    target = TargetTransformer(best_params.get("target_transform", "raw")).fit(y_dev)
    cfg = config_from_dict(best_params, seed=SEED)
    val_model, val_history = train_mlp_fixed(
        Xd, target.transform(y_dev).astype(np.float32), cfg, final_epochs)
    hold_pred = target.inverse_transform(predict_scaled(val_model, Xh))
    holdout_score = rmse(y_hold, hold_pred)

    # Non-MLP benchmark is evaluated only after the MLP selection is frozen.
    ridge_prep = build_preprocessor(X_dev, feature_engineering=True, scaler="standard", skew_threshold=.75)
    ridge_dev = ridge_prep.fit_transform(X_dev)
    ridge = Ridge(alpha=20.0).fit(ridge_dev, np.log1p(y_dev))
    ridge_pred = np.expm1(ridge.predict(ridge_prep.transform(X_hold)))
    benchmark_rmse = rmse(y_hold, ridge_pred)
    pd.DataFrame([{"model": "Ridge benchmark (not candidate)", "holdout_rmse": benchmark_rmse}]).to_csv(
        EXPERIMENTS_DIR / "benchmark.csv", index=False)

    PREDICTIONS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({ID_COLUMN: X_hold[ID_COLUMN].to_numpy(), TARGET: hold_pred}).to_csv(
        PREDICTIONS_DIR / "internal_holdout_predictions.csv", index=False)
    holdout_metrics = {
        "rmse": holdout_score, "mae": float(np.mean(np.abs(y_hold - hold_pred))),
        "bias": float(np.mean(y_hold - hold_pred)), "fixed_epochs": int(final_epochs),
        "ridge_benchmark_rmse": benchmark_rmse,
    }
    (ARTIFACTS_DIR / "holdout_metrics.json").write_text(json.dumps(holdout_metrics, indent=2), encoding="utf-8")
    error_analysis(X_hold, y_hold, hold_pred, df)
    pd.DataFrame(val_history).to_csv(EXPERIMENTS_DIR / "history_validation_fixed.csv", index=False)
    plot_fixed_history(pd.DataFrame(val_history), FIGURES_DIR / "training_final_holdout.png",
                       "Reentrenamiento de validación con épocas fijas")

    validation_metadata = {
        "artifact_role": "operational held-out simulation only", "selection_cv": confirm,
        "holdout_metrics": holdout_metrics, "input_dim": Xd.shape[1], "training_rows": len(X_dev),
    }
    save_artifact(MODELS_DIR / "validation_model", val_model, pre, target, cfg,
                  prep_params(best_params), X.columns, validation_metadata)
    simulated = df.iloc[hold_idx].copy()
    simulated.to_csv(ARTIFACTS_DIR / "simulated_heldout.csv", index=False)

    # Competition artifact retrained on all 1,168 labeled rows using the same
    # epoch count determined only by five-fold development CV.
    metadata = {
        "artifact_role": "competition final model", "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_cv": confirm, "honest_internal_holdout": holdout_metrics,
        "data_shape": list(df.shape), "seed": SEED,
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "mps_available": bool(torch.backends.mps.is_available()), "training_device": "cpu"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "torch": torch.__version__, "optuna": optuna.__version__},
    }
    final_model, final_pre, final_target, final_history = fit_and_save(
        X, y.to_numpy(), best_params, final_epochs, MODELS_DIR / "final", metadata)

    final_row_metrics = {**confirm, "holdout_rmse": holdout_score}
    record_result(results, "final_retrain_all_data", "final", best_params, final_row_metrics,
                  notes=f"Retrained on all rows for {final_epochs} CV-derived epochs")
    pd.DataFrame(results).to_csv(EXPERIMENTS_DIR / "results.csv", index=False)
    pd.DataFrame(final_history).to_csv(EXPERIMENTS_DIR / "history_final_retrain.csv", index=False)
    (ARTIFACTS_DIR / "best_configuration.json").write_text(json.dumps({
        "params": best_params, "confirmatory_cv": confirm, "holdout": holdout_metrics,
        "final_epochs": final_epochs,
    }, indent=2, default=jsonable), encoding="utf-8")
    print(json.dumps({"best_params": best_params, "confirmatory_cv": confirm,
                      "holdout": holdout_metrics, "final_epochs": final_epochs}, indent=2, default=jsonable))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA-independent MLP experiment and final training pipeline")
    parser.add_argument("--trials", type=int, default=18, help="Total number of Optuna trials")
    args = parser.parse_args()
    main(args.trials)
