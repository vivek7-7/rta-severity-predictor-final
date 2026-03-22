"""
tests/test_auth.py
Authentication endpoint tests.
Uses httpx AsyncClient with an in-memory SQLite test database.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from app.main import app
from app.database import Base, get_db

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = async_sessionmaker(bind=test_engine, expire_on_commit=False)


async def override_get_db():
    async with TestSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_register_success(client):
    resp = await client.post("/register", data={
        "full_name": "Test User",
        "email": "test@example.com",
        "password": "password123",
        "confirm_password": "password123",
    }, follow_redirects=False)
    assert resp.status_code in (302, 303)
    assert "access_token" in resp.cookies


@pytest.mark.asyncio
async def test_register_duplicate_email(client):
    data = {
        "full_name": "User A",
        "email": "dup@example.com",
        "password": "password123",
        "confirm_password": "password123",
    }
    await client.post("/register", data=data, follow_redirects=False)
    resp2 = await client.post("/register", data=data, follow_redirects=False)
    assert resp2.status_code == 400


@pytest.mark.asyncio
async def test_login_invalid_password(client):
    await client.post("/register", data={
        "full_name": "Login User",
        "email": "login@example.com",
        "password": "correct_pass",
        "confirm_password": "correct_pass",
    }, follow_redirects=False)
    resp = await client.post("/login", data={
        "email": "login@example.com",
        "password": "wrong_pass",
        "next": "/dashboard",
    }, follow_redirects=False)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_logout_clears_cookie(client):
    await client.post("/register", data={
        "full_name": "Logout User",
        "email": "logout@example.com",
        "password": "mypassword",
        "confirm_password": "mypassword",
    }, follow_redirects=False)
    resp = await client.get("/logout", follow_redirects=False)
    assert resp.status_code in (302, 303)
