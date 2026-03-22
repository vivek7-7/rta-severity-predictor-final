"""
app/models/prediction.py
SQLAlchemy ORM model for storing prediction history.
Each row captures the full input snapshot, result, SHAP values, and metadata.
"""

from datetime import datetime
from sqlalchemy import String, DateTime, Float, Integer, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Prediction(Base):
    """Stores one prediction event per row."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    # Result
    severity_label: Mapped[str] = mapped_column(String(30), nullable=False)
    severity_code: Mapped[int] = mapped_column(Integer, nullable=False)  # 0, 1, 2
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Probabilities as JSON: {"Slight Injury": 0.7, "Serious Injury": 0.2, "Fatal injury": 0.1}
    probabilities: Mapped[dict] = mapped_column(JSON, nullable=False)

    # SHAP top-10 values as JSON: {"feature_name": float, ...}
    shap_values: Mapped[dict] = mapped_column(JSON, nullable=True)

    # Full input snapshot as JSON (31 fields)
    inputs: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Model used
    model_key: Mapped[str] = mapped_column(String(30), nullable=False, default="xgb")

    # Key fields for quick display (denormalized for fast history query)
    cause_of_accident: Mapped[str] = mapped_column(String(120), nullable=True)
    weather_conditions: Mapped[str] = mapped_column(String(120), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Relationship back to user
    user: Mapped["User"] = relationship("User", back_populates="predictions")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Prediction id={self.id} severity={self.severity_label!r} user_id={self.user_id}>"
