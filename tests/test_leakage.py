"""Leakage assertions for the publication main ML pipeline."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_s06():
    spec = importlib.util.spec_from_file_location(
        "s06", Path(__file__).parent.parent / "src" / "ml" / "s06_ml_data_prep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_features():
    return _load_s06().FEATURES


def _get_banned_features():
    return _load_s06().BANNED_MAIN_ML_FEATURES


def test_banned_main_ml_features_not_in_features():
    """Main ML model must not include timing-sensitive or post-birth features."""
    overlap = sorted(set(_get_features()) & _get_banned_features())
    assert not overlap, f"Banned main-ML features found: {overlap}"


def test_feature_count():
    """Exactly 10 conservative main-model features."""
    features = _get_features()
    assert len(features) == 10, f"Expected 10 features, got {len(features)}: {features}"


def test_no_survey_year_in_features():
    """survey_year would be a label-leaking predictor."""
    assert "survey_year" not in _get_features()


def test_no_country_in_features():
    """country_code in features would enable country-level memorisation."""
    assert "country_code" not in _get_features()


def test_s08b_features_match_s06():
    """s08b_regional_shap FEATURES must exactly match s06_ml_data_prep.FEATURES."""
    s06_features = _get_features()
    spec = importlib.util.spec_from_file_location(
        "s08b", Path(__file__).parent.parent / "src" / "ml" / "s08b_regional_shap.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.FEATURES == s06_features, (
        f"s08b FEATURES differ from s06:\n  s06: {s06_features}\n  s08b: {mod.FEATURES}"
    )


def test_split_parquets_have_no_banned_features():
    """Generated split matrices must match the safe main-model schema."""
    import pyarrow.parquet as pq

    ml_dir = Path(__file__).parent.parent / "output" / "ml"
    expected = _get_features()
    banned = _get_banned_features()
    split_names = ["train", "val_internal", "test_time", "test_space"]
    existing = []
    for split in split_names:
        path = ml_dir / f"X_{split}.parquet"
        if path.exists():
            existing.append((split, path))
    if not existing:
        pytest.skip(
            "Split parquet files not found — run src.ml.s06_ml_data_prep first"
        )
    for split, path in existing:
        cols = pq.ParquetFile(path).schema_arrow.names
        assert cols == expected, f"{path.name} columns differ from FEATURES"
        assert not (set(cols) & banned), f"{path.name} includes banned features"


def test_xgb_model_has_no_banned_features():
    """Saved XGBoost model must be trained on the safe main-model schema."""
    model_path = Path(__file__).parent.parent / "output" / "ml" / "xgb_model.json"
    if not model_path.exists():
        pytest.skip("xgb_model.json not found — run src.ml.s07_xgboost_training first")
    obj = json.loads(model_path.read_text(encoding="utf-8"))
    feature_names = obj.get("learner", {}).get("feature_names", []) or []
    assert feature_names == _get_features(), "xgb_model.json feature schema is stale"
    assert not (set(feature_names) & _get_banned_features()), (
        "xgb_model.json includes banned features"
    )