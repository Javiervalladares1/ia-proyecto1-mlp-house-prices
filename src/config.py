from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
FIGURES_DIR = ROOT / "figures"
ARTIFACTS_DIR = ROOT / "artifacts"
EXPERIMENTS_DIR = ROOT / "experiments"
MODELS_DIR = ROOT / "models"
PREDICTIONS_DIR = ROOT / "predictions"
REPORTS_DIR = ROOT / "reports"

SEED = 42
TARGET = "SalePrice"
ID_COLUMN = "Id"

# Variables numericas cuyo codigo representa una categoria sin distancia metrica.
NUMERIC_AS_CATEGORY = ["MSSubClass", "MoSold"]

# Escalas ordinales que se conservan como categoricas porque el orden ya esta
# representado en las etiquetas y el one-hot evita imponer distancias falsas.
ORDINAL_CATEGORICAL = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual",
    "Functional", "FireplaceQu", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "PavedDrive", "LandSlope",
    "LotShape", "Utilities",
]

for directory in [FIGURES_DIR, ARTIFACTS_DIR, EXPERIMENTS_DIR, MODELS_DIR,
                  PREDICTIONS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

