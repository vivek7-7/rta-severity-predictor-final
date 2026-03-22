"""
app/routers/dashboard.py
Dashboard page: aggregate stats, charts data, recent predictions summary.
"""

import logging
from collections import Counter
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.prediction import Prediction
from app.models.user import User
from app.routers.auth import get_current_user
from app.ml.features import SEVERITY_LABELS, MODEL_REGISTRY
from app.ml.predictor import get_metrics_report, is_demo_mode

logger = logging.getLogger(__name__)
router = APIRouter(tags=["dashboard"])
templates = Jinja2Templates(directory="app/templates")


@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="User dashboard",
    description="Overview metrics, charts, and model performance summary.",
)
async def dashboard_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # ── Aggregate counts ───────────────────────────────────────────────────────
    all_preds_q = await db.execute(
        select(Prediction).where(Prediction.user_id == current_user.id)
    )
    all_preds = all_preds_q.scalars().all()

    total = len(all_preds)
    slight = sum(1 for p in all_preds if p.severity_code == 0)
    serious = sum(1 for p in all_preds if p.severity_code == 1)
    fatal = sum(1 for p in all_preds if p.severity_code == 2)

    # ── Doughnut chart data ────────────────────────────────────────────────────
    severity_chart = {
        "labels": ["Slight Injury", "Serious Injury", "Fatal injury"],
        "data": [slight, serious, fatal],
    }

    # ── Line chart: predictions per day (last 30 days) ─────────────────────────
    today = datetime.utcnow().date()
    thirty_days_ago = today - timedelta(days=29)
    daily_counts: dict[str, int] = {}
    for i in range(30):
        day = (thirty_days_ago + timedelta(days=i)).strftime("%Y-%m-%d")
        daily_counts[day] = 0

    for p in all_preds:
        day_str = p.created_at.date().strftime("%Y-%m-%d")
        if day_str in daily_counts:
            daily_counts[day_str] += 1

    line_chart = {
        "labels": list(daily_counts.keys()),
        "data": list(daily_counts.values()),
    }

    # ── Bar chart: top 5 causes of accident ───────────────────────────────────
    causes = [p.cause_of_accident for p in all_preds if p.cause_of_accident]
    cause_counter = Counter(causes).most_common(5)
    cause_chart = {
        "labels": [c[0] for c in cause_counter],
        "data": [c[1] for c in cause_counter],
    }

    # ── Recent predictions (last 5) ───────────────────────────────────────────
    recent_q = await db.execute(
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(5)
    )
    recent = recent_q.scalars().all()

    # ── Model performance strip (top 3 by weighted F1) ─────────────────────────
    metrics = get_metrics_report()
    top_models = []
    if metrics:
        ranked = sorted(
            metrics.items(),
            key=lambda x: x[1].get("weighted_f1", 0),
            reverse=True,
        )[:3]
        for key, m in ranked:
            display_name = MODEL_REGISTRY.get(key, {}).get("name", key.upper())
            top_models.append({
                "key": key,
                "name": display_name,
                "accuracy": round(m.get("accuracy", 0) * 100, 1),
                "weighted_f1": round(m.get("weighted_f1", 0), 3),
            })

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": current_user,
            "total": total,
            "slight": slight,
            "serious": serious,
            "fatal": fatal,
            "severity_chart": severity_chart,
            "line_chart": line_chart,
            "cause_chart": cause_chart,
            "recent": recent,
            "top_models": top_models,
            "demo_mode": is_demo_mode(),
        },
    )
