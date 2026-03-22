"""
app/routers/history.py
Serves the paginated prediction history page with filters and CSV export.
"""

import csv
import io
import logging
from datetime import datetime, date
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.prediction import Prediction
from app.models.user import User
from app.routers.auth import get_current_user
from app.ml.features import SEVERITY_LABELS, MODEL_REGISTRY

logger = logging.getLogger(__name__)
router = APIRouter(tags=["history"])
templates = Jinja2Templates(directory="app/templates")


@router.get(
    "/history",
    response_class=HTMLResponse,
    summary="Prediction history",
    description="Paginated table of past predictions with severity, confidence, filters.",
)
async def history_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    page: int = Query(default=1, ge=1),
    severity: Optional[str] = Query(default=None),
    model_filter: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
):
    # Build filter conditions
    conditions = [Prediction.user_id == current_user.id]

    if severity and severity in SEVERITY_LABELS.values():
        conditions.append(Prediction.severity_label == severity)

    if model_filter and model_filter in MODEL_REGISTRY:
        conditions.append(Prediction.model_key == model_filter)

    if date_from:
        try:
            dt_from = datetime.strptime(date_from, "%Y-%m-%d")
            conditions.append(Prediction.created_at >= dt_from)
        except ValueError:
            pass

    if date_to:
        try:
            dt_to = datetime.strptime(date_to, "%Y-%m-%d")
            conditions.append(Prediction.created_at <= dt_to)
        except ValueError:
            pass

    # Count total matching rows
    count_q = await db.execute(
        select(func.count()).select_from(Prediction).where(and_(*conditions))
    )
    total = count_q.scalar() or 0
    total_pages = max(1, (total + settings.HISTORY_PAGE_SIZE - 1) // settings.HISTORY_PAGE_SIZE)
    page = min(page, total_pages)

    # Fetch page
    offset = (page - 1) * settings.HISTORY_PAGE_SIZE
    rows_q = await db.execute(
        select(Prediction)
        .where(and_(*conditions))
        .order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(settings.HISTORY_PAGE_SIZE)
    )
    predictions = rows_q.scalars().all()

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "user": current_user,
            "predictions": predictions,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "severity_labels": list(SEVERITY_LABELS.values()),
            "model_registry": MODEL_REGISTRY,
            "filters": {
                "severity": severity or "",
                "model_filter": model_filter or "",
                "date_from": date_from or "",
                "date_to": date_to or "",
            },
        },
    )


@router.get(
    "/history/export",
    summary="Export history as CSV",
    description="Downloads all of the current user's predictions as a CSV file.",
)
async def export_csv(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows_q = await db.execute(
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
    )
    predictions = rows_q.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Timestamp", "Severity", "Severity Code", "Confidence (%)",
        "Cause of Accident", "Weather", "Model Used",
    ])
    for p in predictions:
        writer.writerow([
            p.id,
            p.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            p.severity_label,
            p.severity_code,
            f"{p.confidence * 100:.1f}",
            p.cause_of_accident or "",
            p.weather_conditions or "",
            p.model_key,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=rta_predictions.csv"},
    )


@router.post(
    "/history/{pred_id}/delete",
    summary="Delete a prediction",
    description="Deletes a prediction record owned by the current user.",
)
async def delete_prediction(
    pred_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from fastapi import HTTPException
    result = await db.execute(
        select(Prediction).where(
            Prediction.id == pred_id,
            Prediction.user_id == current_user.id,
        )
    )
    pred = result.scalar_one_or_none()
    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    await db.delete(pred)
    await db.commit()
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/history", status_code=status.HTTP_303_SEE_OTHER)
