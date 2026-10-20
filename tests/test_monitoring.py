"""PSI проверяем на распределениях, где ответ известен заранее."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from monitoring import drift_report, psi, status  # noqa: E402

rng = np.random.default_rng(0)


def test_same_distribution_is_stable():
    assert psi(rng.normal(0, 1, 20_000), rng.normal(0, 1, 20_000)) < 0.02


def test_shifted_distribution_is_drift():
    assert psi(rng.normal(0, 1, 20_000), rng.normal(1, 1, 20_000)) > 0.25


def test_constant_feature_and_missing_values():
    assert psi(np.ones(100), np.ones(100)) < 0.01
    assert psi(np.ones(100), np.zeros(100)) > 0.25
    assert psi(np.array([1.0, np.nan, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, np.nan])) < 0.5


def test_report_sorted_with_status():
    base = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(0, 1, 5000)})
    new = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(2, 1, 5000)})
    rep = drift_report(base, new, ["a", "b"])
    assert list(rep.feature) == ["b", "a"]
    assert rep.status.tolist() == [status(rep.psi[0]), "стабильно"]
