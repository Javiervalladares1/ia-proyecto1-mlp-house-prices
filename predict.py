from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ID_COLUMN, MODELS_DIR, PREDICTIONS_DIR, TARGET
from src.model import load_checkpoint, predict_scaled


def predict_file(input_csv: Path, output_csv: Path, model_dir: Path):
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    expected = metadata["expected_columns"]
    df = pd.read_csv(input_csv)
    has_target = TARGET in df.columns
    y_true = df[TARGET].to_numpy() if has_target else None
    X = df.drop(columns=[TARGET], errors="ignore")
    missing = sorted(set(expected) - set(X.columns))
    if missing:
        raise ValueError(f"Faltan {len(missing)} columnas requeridas: {missing}")
    extra = sorted(set(X.columns) - set(expected))
    if extra:
        print(f"Aviso: se ignoraran columnas adicionales: {extra}")
    X = X[expected]
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    target = joblib.load(model_dir / "target_transformer.joblib")
    model, _, input_dim = load_checkpoint(model_dir / "model.pt", device="cpu")
    Xp = preprocessor.transform(X).astype(np.float32)
    if Xp.shape[1] != input_dim:
        raise RuntimeError(f"Dimension procesada inesperada: {Xp.shape[1]} != {input_dim}")
    pred = target.inverse_transform(predict_scaled(model, Xp, device="cpu"))
    ids = X[ID_COLUMN].to_numpy() if ID_COLUMN in X else np.arange(1, len(X) + 1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COLUMN: ids, TARGET: pred}).to_csv(output_csv, index=False)
    print(f"Predicciones guardadas: {output_csv} ({len(pred)} filas)")
    if has_target:
        score = float(np.sqrt(np.mean((y_true - pred) ** 2)))
        print(f"RMSE en escala original de SalePrice: {score:,.2f}")
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera predicciones con el MLP final guardado")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=PREDICTIONS_DIR / "predictions.csv")
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR / "final")
    args = parser.parse_args()
    predict_file(args.input_csv, args.output, args.model_dir)

