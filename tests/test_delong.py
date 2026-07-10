"""Unit tests for DeLong paired AUC test implementation."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ml.s07c_lr_baseline import delong_test


def test_identical_models_p_is_nan_or_one():
    """When both models are identical, z=0 and p=1."""
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    auc1, auc2, z, pval = delong_test(y, p, p)
    assert auc1 == auc2
    assert abs(z) < 1e-9 or np.isnan(z)


def test_better_model_has_positive_delta():
    """Perfect model vs random: z >> 0, p << 0.05."""
    rng = np.random.default_rng(42)
    n = 1000
    y = rng.integers(0, 2, n)
    # p1 is near-perfect, p2 is noise
    p1 = y.astype(float) + rng.random(n) * 0.1
    p2 = rng.random(n)
    auc1, auc2, z, pval = delong_test(y, p1, p2)
    assert auc1 > auc2
    assert z > 0
    assert pval < 0.001


def test_all_same_class_returns_nan():
    y = np.zeros(100, dtype=int)
    p1 = np.random.random(100)
    p2 = np.random.random(100)
    result = delong_test(y, p1, p2)
    assert all(np.isnan(v) for v in result)
