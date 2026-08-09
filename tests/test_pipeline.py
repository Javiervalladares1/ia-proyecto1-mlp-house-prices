import tempfile
import unittest
from pathlib import Path

import pandas as pd

from predict import predict_file
from src.config import DATA_PATH, MODELS_DIR


class CompetitionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.sample = pd.read_csv(DATA_PATH).head(5).drop(columns=["SalePrice"])

    def test_unseen_category_and_column_order(self):
        sample = self.sample.copy()
        sample.loc[0, "MSZoning"] = "__UNSEEN_CATEGORY__"
        sample = sample[sample.columns[::-1]]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "output.csv"
            sample.to_csv(input_path, index=False)
            pred = predict_file(input_path, output_path, MODELS_DIR / "final")
            result = pd.read_csv(output_path)
            self.assertEqual(len(pred), 5)
            self.assertEqual(result.columns.tolist(), ["Id", "SalePrice"])
            self.assertTrue((result["SalePrice"] > 0).all())

    def test_missing_required_column_fails_clearly(self):
        sample = self.sample.drop(columns=["OverallQual"])
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "output.csv"
            sample.to_csv(input_path, index=False)
            with self.assertRaisesRegex(ValueError, "OverallQual"):
                predict_file(input_path, output_path, MODELS_DIR / "final")


if __name__ == "__main__":
    unittest.main()

