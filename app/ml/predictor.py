"""
app/ml/predictor.py
Loads artifacts, runs inference, computes SHAP.
Falls back to demo mode if artifacts missing or incompatible.
"""
import logging, random
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from app.ml.features import (
    FEATURE_ORDER, FEATURE_DISPLAY, SEVERITY_LABELS,
    ARTIFACTS_DIR, MODEL_REGISTRY
)

logger = logging.getLogger(__name__)

_models: dict[str, Any] = {}
_scaler = None
_encoders: dict[str, Any] = {}
_shap_explainer = None
_metrics_report: dict = {}
_demo_mode = False


def load_artifacts() -> None:
    global _scaler, _encoders, _shap_explainer, _metrics_report, _demo_mode
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    missing = []

    # ── Scaler ────────────────────────────────────────────────────────────────
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    if scaler_path.exists():
        try:
            _scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler.pkl")
        except Exception as e:
            logger.error("Failed to load scaler.pkl: %s", e)
            missing.append("scaler.pkl")
    else:
        missing.append("scaler.pkl")

    # ── Encoders ──────────────────────────────────────────────────────────────
    encoders_path = ARTIFACTS_DIR / "encoders.pkl"
    if encoders_path.exists():
        try:
            _encoders = joblib.load(encoders_path)
            logger.info("Loaded encoders.pkl (%d encoders)", len(_encoders))
        except Exception as e:
            logger.error("Failed to load encoders.pkl: %s", e)
            missing.append("encoders.pkl")
    else:
        missing.append("encoders.pkl")

    # ── Models — load each individually, skip failures ────────────────────────
    for key, info in MODEL_REGISTRY.items():
        p: Path = info["file"]
        if p.exists():
            try:
                _models[key] = joblib.load(p)
                logger.info("Loaded model: %s", key)
            except Exception as e:
                logger.warning("Skipping model %s — load failed: %s", key, e)
                missing.append(p.name)
        else:
            missing.append(p.name)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_path = ARTIFACTS_DIR / "shap_explainer.pkl"
    if shap_path.exists():
        try:
            _shap_explainer = joblib.load(shap_path)
            logger.info("Loaded shap_explainer.pkl")
        except Exception as e:
            logger.warning("SHAP explainer skipped: %s", e)

    # ── Metrics report ────────────────────────────────────────────────────────
    import json
    metrics_path = ARTIFACTS_DIR / "metrics_report.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            _metrics_report = json.load(f)
        logger.info("Loaded metrics_report.json")

    # ── Decide demo mode ──────────────────────────────────────────────────────
    # Demo mode only if scaler OR encoders missing — models can be partial
    critical_missing = [m for m in missing if m in ("scaler.pkl", "encoders.pkl")]
    if critical_missing or len(_models) == 0:
        logger.warning("Critical artifacts missing: %s — DEMO MODE", critical_missing)
        _demo_mode = True
    else:
        _demo_mode = False
        logger.info("Predictor ready. Loaded %d models.", len(_models))
        if missing:
            logger.warning("Some models skipped (version mismatch): %s", missing)


def _encode_inputs(raw_inputs: dict) -> np.ndarray:
    row = []
    for feature in FEATURE_ORDER:
        value = (raw_inputs.get(feature)
                 or raw_inputs.get(feature.lower(), "Unknown"))
        encoder = _encoders.get(feature.lower())
        if encoder is not None:
            try:
                encoded = encoder.transform([str(value)])[0]
            except (ValueError, AttributeError):
                encoded = 0
        else:
            try:
                encoded = float(value)
            except (ValueError, TypeError):
                encoded = 0.0
        row.append(encoded)
    X = np.array(row, dtype=float).reshape(1, -1)
    if _scaler is not None:
        X = _scaler.transform(X)
    return X


def _compute_shap(X: np.ndarray, predicted_class: int) -> dict:
    if _shap_explainer is None:
        values = {
            FEATURE_DISPLAY[f]: round(random.uniform(-0.25, 0.25), 4)
            for f in FEATURE_ORDER
        }
        return dict(sorted(values.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
    try:
        shap_vals = _shap_explainer.shap_values(X)
        if isinstance(shap_vals, list):
            class_shap = shap_vals[predicted_class][0]
        else:
            class_shap = shap_vals[0]
        values = {
            FEATURE_DISPLAY[FEATURE_ORDER[i]]: round(float(class_shap[i]), 4)
            for i in range(len(FEATURE_ORDER))
        }
        return dict(sorted(values.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
    except Exception as e:
        logger.exception("SHAP failed: %s", e)
        return {}


def predict(raw_inputs: dict, model_key: str = "gb") -> dict:
    """Run prediction. Falls back to demo if model unavailable."""
    if _demo_mode or model_key not in _models:
        # Try any available model before going full demo
        if _models and not _demo_mode:
            fallback_key = next(iter(_models))
            logger.warning("Model %s unavailable, using %s", model_key, fallback_key)
            model_key = fallback_key
        else:
            return _demo_predict(model_key)

    model = _models[model_key]
    X = _encode_inputs(raw_inputs)

    try:
        proba = model.predict_proba(X)[0]
        predicted_class = int(np.argmax(proba))
        confidence = float(np.max(proba))
    except AttributeError:
        raw_pred = float(model.predict(X)[0])
        predicted_class = 0 if raw_pred < 0.5 else (1 if raw_pred < 1.5 else 2)
        proba = [0.0, 0.0, 0.0]
        proba[predicted_class] = 1.0
        confidence = 1.0

    return {
        "severity_label": SEVERITY_LABELS[predicted_class],
        "severity_code": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {SEVERITY_LABELS[i]: round(float(proba[i]), 4) for i in range(3)},
        "shap_values": _compute_shap(X, predicted_class),
        "model_key": model_key,
    }


def _demo_predict(model_key: str = "gb") -> dict:
    predicted_class = random.choices([0, 1, 2], weights=[75, 20, 5])[0]
    raw = [random.uniform(0.01, 0.99) for _ in range(3)]
    total = sum(raw)
    proba = [round(p / total, 4) for p in raw]
    shap_values = {
        FEATURE_DISPLAY[f]: round(random.uniform(-0.25, 0.25), 4)
        for f in FEATURE_ORDER
    }
    return {
        "severity_label": SEVERITY_LABELS[predicted_class],
        "severity_code": predicted_class,
        "confidence": round(max(proba), 4),
        "probabilities": {SEVERITY_LABELS[i]: proba[i] for i in range(3)},
        "shap_values": dict(
            sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        ),
        "model_key": model_key,
    }


def get_metrics_report() -> dict: return _metrics_report
def is_demo_mode() -> bool: return _demo_mode
def get_loaded_models() -> list: return list(_models.keys())
