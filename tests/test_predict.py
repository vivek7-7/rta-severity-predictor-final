"""
tests/test_predict.py
Prediction flow tests — uses demo mode (no real .pkl files required).
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.database import Base, get_db
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DB_URL)
TestSession = async_sessionmaker(bind=test_engine, expire_on_commit=False)


async def override_get_db():
    async with TestSession() as s:
        yield s


app.dependency_overrides[get_db] = override_get_db

FORM_DATA = {
    "day_of_week": "Monday",
    "age_band_of_driver": "18-30",
    "sex_of_driver": "Male",
    "educational_level": "High school",
    "vehicle_driver_relation": "Owner",
    "driving_experience": "2-5yr",
    "type_of_vehicle": "Automobile",
    "owner_of_vehicle": "Owner",
    "service_year_of_vehicle": "2-5yr",
    "defect_of_vehicle": "No defect",
    "area_accident_occured": "Residential areas",
    "lanes_or_medians": "Undivided Two way",
    "road_allignment": "Tangent road with flat terrain",
    "types_of_junction": "No junction",
    "road_surface_type": "Asphalt roads",
    "road_surface_conditions": "Dry",
    "light_conditions": "Daylight",
    "weather_conditions": "Normal",
    "type_of_collision": "Vehicle with vehicle collision",
    "number_of_vehicles_involved": "2",
    "number_of_casualties": "1",
    "vehicle_movement": "Going straight",
    "casualty_class": "Driver or rider",
    "sex_of_casualty": "Male",
    "age_band_of_casualty": "18-30",
    "casualty_severity": "3",
    "work_of_casuality": "Driver",
    "fitness_of_casuality": "Normal",
    "pedestrian_movement": "Not a Pedestrian",
    "cause_of_accident": "Overspeed",
    "hour_of_day": "14",
    "model_key": "xgb",
}


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def auth_client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        # Register + login
        await c.post("/register", data={
            "full_name": "Predict User",
            "email": "predict@test.com",
            "password": "testpass123",
            "confirm_password": "testpass123",
        }, follow_redirects=False)
        yield c


@pytest.mark.asyncio
async def test_predict_form_requires_auth():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        resp = await c.get("/predict", follow_redirects=False)
        assert resp.status_code in (307, 302, 303)


@pytest.mark.asyncio
async def test_predict_submit_redirects_to_result(auth_client):
    resp = await auth_client.post("/predict", data=FORM_DATA, follow_redirects=False)
    assert resp.status_code in (302, 303)
    assert "/result/" in resp.headers.get("location", "")


@pytest.mark.asyncio
async def test_result_page_loads(auth_client):
    redirect = await auth_client.post("/predict", data=FORM_DATA, follow_redirects=False)
    location = redirect.headers["location"]
    resp = await auth_client.get(location, follow_redirects=False)
    assert resp.status_code == 200
    assert b"Predicted Severity" in resp.content
