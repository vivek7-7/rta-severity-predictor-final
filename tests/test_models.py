"""
tests/test_models.py
Unit tests for the ML predictor module and feature definitions.
Runs entirely in demo mode — no .pkl files required.
"""

import pytest
from app.ml import predictor
from app.ml.features import (
    FEATURE_ORDER, FEATURE_OPTIONS, FEATURE_DISPLAY,
    SEVERITY_LABELS, MODEL_REGISTRY
)


# ── Feature definition tests ───────────────────────────────────────────────────
def test_feature_order_length():
    assert len(FEATURE_ORDER) == 31


def test_all_features_have_display_names():
    for f in FEATURE_ORDER:
        assert f in FEATURE_DISPLAY, f"Missing display name for: {f}"


def test_all_features_have_options():
    for f in FEATURE_ORDER:
        assert f in FEATURE_OPTIONS, f"Missing options for: {f}"
        assert len(FEATURE_OPTIONS[f]) > 0, f"Empty options for: {f}"


def test_severity_labels():
    assert SEVERITY_LABELS[0] == "Slight Injury"
    assert SEVERITY_LABELS[1] == "Serious Injury"
    assert SEVERITY_LABELS[2] == "Fatal injury"


def test_model_registry_has_all_keys():
    expected = {"xgb", "rf", "lgbm", "gb", "svm", "lr", "dt", "knn", "nb", "mlp", "ridge", "lasso"}
    assert expected.issubset(set(MODEL_REGISTRY.keys()))


def test_xgb_is_default():
    assert MODEL_REGISTRY["xgb"]["default"] is True
    non_defaults = [k for k, v in MODEL_REGISTRY.items() if v.get("default") and k != "xgb"]
    assert len(non_defaults) == 0


# ── Predictor demo mode tests ──────────────────────────────────────────────────
def test_demo_predict_returns_valid_structure():
    result = predictor._demo_predict("xgb")
    assert "severity_label" in result
    assert "severity_code" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "shap_values" in result
    assert result["severity_code"] in (0, 1, 2)
    assert 0.0 <= result["confidence"] <= 1.0


def test_demo_predict_probabilities_sum_to_one():
    result = predictor._demo_predict("xgb")
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 0.01


def test_demo_predict_shap_top10():
    result = predictor._demo_predict("xgb")
    assert len(result["shap_values"]) <= 10


def test_predict_fallback_to_demo_when_no_artifacts():
    """Without loading artifacts, predict() should return demo results."""
    sample_input = {f: FEATURE_OPTIONS[f][0] for f in FEATURE_ORDER}
    result = predictor.predict(sample_input, model_key="xgb")
    assert result["severity_label"] in SEVERITY_LABELS.values()
    assert result["severity_code"] in (0, 1, 2)


def test_predict_unknown_model_key_uses_demo():
    sample_input = {f: FEATURE_OPTIONS[f][0] for f in FEATURE_ORDER}
    result = predictor.predict(sample_input, model_key="nonexistent_model")
    assert result["model_key"] == "nonexistent_model"


def test_get_metrics_report_returns_dict():
    report = predictor.get_metrics_report()
    assert isinstance(report, dict)


def test_is_demo_mode_bool():
    assert isinstance(predictor.is_demo_mode(), bool)


def test_get_loaded_models_returns_list():
    models = predictor.get_loaded_models()
    assert isinstance(models, list)
