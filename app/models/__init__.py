"""app/models/__init__.py — ORM model package."""
from app.models.user import User
from app.models.prediction import Prediction

__all__ = ["User", "Prediction"]
