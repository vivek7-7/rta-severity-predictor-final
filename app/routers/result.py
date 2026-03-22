"""
app/routers/result.py
Serves the /result/{id} page showing SHAP, confidence ring, probability bars,
and key input summary for a specific prediction.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.prediction import Prediction
from app.models.user import User
from app.routers.auth import get_current_user
from app.ml.features import SEVERITY_COLORS, FEATURE_DISPLAY, MODEL_REGISTRY

logger = logging.getLogger(__name__)
router = APIRouter(tags=["result"])
templates = Jinja2Templates(directory="app/templates")

# Top 8 most informative fields to surface in the key inputs summary
KEY_INPUT_FIELDS = [
    "cause_of_accident",
    "driving_experience",
    "type_of_collision",
    "weather_conditions",
    "light_conditions",
    "road_surface_conditions",
    "age_band_of_driver",
    "vehicle_movement",
]


@router.get(
    "/result/{pred_id}",
    response_class=HTMLResponse,
    summary="Prediction result page",
    description="Shows confidence ring, SHAP waterfall, probability bars, and key inputs.",
)
async def result_page(
    pred_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Prediction).where(
            Prediction.id == pred_id,
            Prediction.user_id == current_user.id,
        )
    )
    pred = result.scalar_one_or_none()
    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    severity_color = SEVERITY_COLORS.get(pred.severity_label, "gray")

    # Sort SHAP values for display (desc by absolute magnitude)
    shap_sorted = {}
    if pred.shap_values:
        shap_sorted = dict(
            sorted(pred.shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        )

    # Key inputs summary
    key_inputs = {
        FEATURE_DISPLAY.get(k, k): pred.inputs.get(k, "—")
        for k in KEY_INPUT_FIELDS
        if k in pred.inputs
    }

    model_name = MODEL_REGISTRY.get(pred.model_key, {}).get("name", pred.model_key.upper())

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "user": current_user,
            "pred": pred,
            "pred_id": pred.id,
            "severity_label": pred.severity_label,
            "severity_color": severity_color,
            "confidence": pred.confidence,
            "probabilities": pred.probabilities,
            "shap_values": shap_sorted,
            "inputs": pred.inputs,
            "key_inputs": key_inputs,
            "model_name": model_name,
            "timestamp": pred.created_at,
        },
    )
