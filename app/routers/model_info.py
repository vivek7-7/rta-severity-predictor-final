"""app/routers/model_info.py"""
import logging
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.models.user import User
from app.routers.auth import get_current_user
from app.ml.features import MODEL_REGISTRY, SEVERITY_LABELS
from app.ml.predictor import get_metrics_report, is_demo_mode

logger = logging.getLogger(__name__)
router = APIRouter(tags=["model_info"])
templates = Jinja2Templates(directory="app/templates")

DATASET_INFO = {
    "name": "Road Traffic Accidents — Addis Ababa Sub-City",
    "source": "Kaggle / Addis Ababa Sub-City Police Dept.",
    "rows": 12316, "features": 25,
    "classes": ["Slight Injury", "Serious Injury", "Fatal injury"],
    "class_distribution": {"Slight Injury": 84.7, "Serious Injury": 14.2, "Fatal injury": 1.1},
    "imbalance_strategy": "SMOTE", "missing_values": "Mode imputation per column",
}

EDA = {
    "day_labels":    ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
    "day_counts":    [1623,1580,1710,1690,1924,1820,969],
    "cause_labels":  ["No distancing","Moving Backward","Overspeed","No priority veh.","Carelessly"],
    "cause_counts":  [2847,1923,1456,1234,987],
    "light_labels":  ["Daylight","Dark-lit","Dark-no light","Dark-unlit"],
    "light_slight":  [85.1,83.8,71.9,82.5],
    "light_serious": [13.9,14.2,25.5,17.5],
    "light_fatal":   [1.0,2.0,2.6,0.0],
}

@router.get("/model-info", response_class=HTMLResponse)
async def model_info_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    selected_model: str = Query(default="gb"),
    compare_a: str = Query(default="gb"),
    compare_b: str = Query(default="rf"),
):
    metrics = get_metrics_report()

    rows = []
    for key, info in MODEL_REGISTRY.items():
        m = metrics.get(key, {})
        fr = m.get("recall_per_class", {}).get("Fatal injury", 0) if m else 0
        rows.append({
            "key": key, "name": info["name"],
            "unit": info["unit"], "type": info["type"],
            "accuracy":    round(m.get("accuracy",0)*100, 2) if m else 0,
            "weighted_f1": round(m.get("weighted_f1",0), 3) if m else 0,
            "macro_f1":    round(m.get("macro_f1",0), 3) if m else 0,
            "roc_auc":     round(m.get("roc_auc",0), 3) if m else 0,
            "fatal_recall":round(fr*100, 1),
            "train_time":  round(m.get("train_time_seconds",0), 1) if m else 0,
            "is_default":  info.get("default", False),
        })

    selected_cm = []
    if metrics and selected_model in metrics:
        selected_cm = metrics[selected_model].get("confusion_matrix", [])

    feat_imp = {}
    if metrics and selected_model in metrics:
        feat_imp = metrics[selected_model].get("feature_importance", {})
    if not feat_imp and metrics:
        for k in ["gb","rf","xgb","lgbm"]:
            if k in metrics and metrics[k].get("feature_importance"):
                feat_imp = metrics[k]["feature_importance"]; break

    def model_stats(key):
        m = metrics.get(key, {})
        if not m: return {}
        fr = m.get("recall_per_class",{}).get("Fatal injury",0)
        return {
            "name": MODEL_REGISTRY.get(key,{}).get("name",key),
            "accuracy":    round(m.get("accuracy",0)*100,2),
            "weighted_f1": round(m.get("weighted_f1",0),3),
            "macro_f1":    round(m.get("macro_f1",0),3),
            "roc_auc":     round(m.get("roc_auc",0),3),
            "fatal_recall":round(fr*100,1),
            "train_time":  round(m.get("train_time_seconds",0),1),
            "cm":          m.get("confusion_matrix",[]),
        }

    return templates.TemplateResponse("model_info.html", {
        "request": request, "user": current_user,
        "dataset_info": DATASET_INFO, "eda": EDA,
        "comparison_rows": rows,
        "selected_model": selected_model,
        "selected_model_name": MODEL_REGISTRY.get(selected_model,{}).get("name",selected_model),
        "selected_cm": selected_cm,
        "feature_importance": feat_imp,
        "severity_labels": list(SEVERITY_LABELS.values()),
        "demo_mode": is_demo_mode(), "has_metrics": bool(metrics),
        "model_registry": MODEL_REGISTRY,
        "compare_a": compare_a, "compare_b": compare_b,
        "compare_a_data": model_stats(compare_a),
        "compare_b_data": model_stats(compare_b),
        "chart_names": [r["name"] for r in rows],
        "chart_f1":    [r["weighted_f1"] for r in rows],
        "chart_acc":   [r["accuracy"] for r in rows],
        "chart_fatal": [r["fatal_recall"] for r in rows],
    })
