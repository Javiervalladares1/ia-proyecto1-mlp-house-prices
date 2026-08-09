from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from .config import ID_COLUMN, NUMERIC_AS_CATEGORY, TARGET


class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic, row-wise domain features; it learns no dataset statistics."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        if not self.enabled:
            return X

        def total(name: str, columns: Iterable[str]):
            available = [c for c in columns if c in X]
            if available:
                X[name] = X[available].fillna(0).sum(axis=1)

        total("TotalSF", ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GarageArea", "WoodDeckSF", "OpenPorchSF"])
        total("TotalBathrooms", ["FullBath", "BsmtFullBath"])
        if {"HalfBath", "BsmtHalfBath"}.intersection(X.columns):
            X["TotalBathrooms"] = X.get("TotalBathrooms", 0) + 0.5 * X[[c for c in ["HalfBath", "BsmtHalfBath"] if c in X]].fillna(0).sum(axis=1)
        total("TotalPorchSF", ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"])
        total("TotalHomeQuality", ["OverallQual", "OverallCond"])
        total("TotalRooms", ["TotRmsAbvGrd", "BedroomAbvGr", "KitchenAbvGr"])

        if "YrSold" in X and "YearBuilt" in X:
            X["HouseAgeAtSale"] = (X["YrSold"] - X["YearBuilt"]).clip(lower=0)
        if "YrSold" in X and "YearRemodAdd" in X:
            X["YearsSinceRemodel"] = (X["YrSold"] - X["YearRemodAdd"]).clip(lower=0)
        if "YrSold" in X and "GarageYrBlt" in X:
            X["GarageAgeAtSale"] = (X["YrSold"] - X["GarageYrBlt"]).clip(lower=0)
        if "YearRemodAdd" in X and "YearBuilt" in X:
            X["WasRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(float)
        if "OverallQual" in X and "GrLivArea" in X:
            X["Qual_x_GrLivArea"] = X["OverallQual"] * X["GrLivArea"]
        if "OverallQual" in X and "TotalSF" in X:
            X["Qual_x_TotalSF"] = X["OverallQual"] * X["TotalSF"]
        return X


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, quantile: float | None = None):
        self.quantile = quantile

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if self.quantile is None:
            self.lower_ = self.upper_ = None
        else:
            self.lower_ = np.nanquantile(X, self.quantile, axis=0)
            self.upper_ = np.nanquantile(X, 1 - self.quantile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.lower_ is None:
            return X
        return np.clip(X, self.lower_, self.upper_)


class SkewLogTransformer(BaseEstimator, TransformerMixin):
    """Learns on the training fold which nonnegative columns merit log1p."""

    def __init__(self, threshold: float | None = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if self.threshold is None:
            self.mask_ = np.zeros(X.shape[1], dtype=bool)
            return self
        centered = X - np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        denom = np.where(std > 0, std ** 3, np.inf)
        skew = np.nanmean(centered ** 3, axis=0) / denom
        self.mask_ = (np.nanmin(X, axis=0) >= 0) & (np.abs(skew) > self.threshold)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        X[:, self.mask_] = np.log1p(np.maximum(X[:, self.mask_], 0))
        return X


@dataclass
class TargetTransformer:
    mode: str = "raw"
    mean_: float | None = None
    std_: float | None = None
    min_: float | None = None
    max_: float | None = None

    def fit(self, y):
        original = np.asarray(y, dtype=np.float64)
        self.min_ = float(original.min())
        self.max_ = float(original.max())
        values = self._forward(original)
        self.mean_ = float(values.mean())
        self.std_ = float(values.std()) or 1.0
        return self

    def _forward(self, y):
        if self.mode == "log1p":
            return np.log1p(y)
        if self.mode != "raw":
            raise ValueError(f"Transformacion objetivo desconocida: {self.mode}")
        return y

    def transform(self, y):
        return (self._forward(np.asarray(y, dtype=np.float64)) - self.mean_) / self.std_

    def inverse_transform(self, y):
        values = np.asarray(y, dtype=np.float64) * self.std_ + self.mean_
        if self.mode == "log1p":
            values = np.expm1(values)
        # A fold-derived guardrail prevents exponentiation from turning a rare
        # extrapolation into a multi-million-dollar squared-error catastrophe.
        return np.clip(values, 0, self.max_ * 1.5)


def split_feature_types(X: pd.DataFrame, feature_engineering: bool = True):
    sample = AmesFeatureEngineer(feature_engineering).fit_transform(X)
    categorical = list(sample.select_dtypes(exclude=np.number).columns)
    categorical += [c for c in NUMERIC_AS_CATEGORY if c in sample and c not in categorical]
    numeric = [c for c in sample.columns if c not in categorical and c not in [TARGET, ID_COLUMN]]
    categorical = [c for c in categorical if c not in [TARGET, ID_COLUMN]]
    return numeric, categorical


def build_preprocessor(
    X: pd.DataFrame,
    scaler: str = "standard",
    skew_threshold: float | None = 0.75,
    clip_quantile: float | None = None,
    min_frequency: int | None = None,
    feature_engineering: bool = True,
):
    numeric, categorical = split_feature_types(X, feature_engineering)
    scaler_obj = StandardScaler() if scaler == "standard" else RobustScaler()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("clip", QuantileClipper(clip_quantile)),
        ("skew_log", SkewLogTransformer(skew_threshold)),
        ("scale", scaler_obj),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency,
                                 sparse_output=False, dtype=np.float32)),
    ])
    columns = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=False)
    return Pipeline([
        ("feature_engineering", AmesFeatureEngineer(feature_engineering)),
        ("columns", columns),
    ])
