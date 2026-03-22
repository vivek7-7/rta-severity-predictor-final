"""
app/schemas/prediction.py
Pydantic schemas for prediction input validation and result output.
All 31 feature fields are validated as non-empty strings.
"""

from pydantic import BaseModel, field_validator, ConfigDict
from typing import Optional
from datetime import datetime


class PredictionInput(BaseModel):
    """Validated input for a single prediction (all 31 features)."""

    # Suppress Pydantic v2 protected namespace warning for model_key field
    model_config = ConfigDict(protected_namespaces=())

    day_of_week: str
    age_band_of_driver: str
    sex_of_driver: str
    educational_level: str
    vehicle_driver_relation: str
    driving_experience: str
    type_of_vehicle: str
    owner_of_vehicle: str
    service_year_of_vehicle: str
    defect_of_vehicle: str
    area_accident_occured: str
    lanes_or_medians: str
    road_allignment: str
    types_of_junction: str
    road_surface_type: str
    road_surface_conditions: str
    light_conditions: str
    weather_conditions: str
    type_of_collision: str
    number_of_vehicles_involved: str
    number_of_casualties: str
    vehicle_movement: str
    casualty_class: str
    sex_of_casualty: str
    age_band_of_casualty: str
    casualty_severity: str
    work_of_casuality: str
    fitness_of_casuality: str
    pedestrian_movement: str
    cause_of_accident: str
    hour_of_day: str

    model_key: str = "xgb"

    @field_validator("*", mode="before")
    @classmethod
    def not_empty(cls, v):
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("Field cannot be empty.")
        return v


class PredictionResult(BaseModel):
    """Full prediction result returned to the UI."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    severity_label: str
    severity_code: int
    confidence: float
    probabilities: dict[str, float]
    shap_values: Optional[dict[str, float]]
    inputs: dict
    model_key: str
    created_at: datetime
