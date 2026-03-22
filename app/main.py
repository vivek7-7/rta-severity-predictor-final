"""
app/main.py
FastAPI application entry point.
- Registers all routers
- Loads ML artifacts on startup via lifespan context manager
- Mounts static files
- Redirects root / to /dashboard
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import create_tables
from app.ml import predictor
from app.routers import auth, predict, result, history, dashboard, model_info

# Ensure required directories exist (fixes Render deploy crash on empty folders)
Path("app/static").mkdir(parents=True, exist_ok=True)
Path("app/ml/artifacts").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB tables + load ML artifacts. Shutdown: nothing to clean."""
    logger.info("▶  Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    await create_tables()
    predictor.load_artifacts()
    logger.info("✓  Application ready.")
    yield
    logger.info("■  Application shutting down.")


# Add enumerate filter for Jinja2 templates
def _enumerate(iterable, start=0):
    return enumerate(iterable, start)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Predicts road accident severity (Slight / Serious / Fatal) "
        "using the Addis Ababa RTA dataset. Trained with 12 ML algorithms "
        "covering the full university syllabus (Units I–V)."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ───────────────────────────────────────────────────────────────
# Create the static directory if it doesn't exist (needed on Render first deploy)
import os as _os
_os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ── Routers ────────────────────────────────────────────────────────────────────
# Register Jinja2 filters
from fastapi.templating import Jinja2Templates as _T
import app.routers.auth as _auth
_auth.templates.env.filters["enumerate"] = _enumerate

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(result.router)
app.include_router(history.router)
app.include_router(dashboard.router)
app.include_router(model_info.router)


# ── Root redirect ──────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/dashboard")


# ── Global exception handler for auth redirects ────────────────────────────────
@app.exception_handler(307)
async def redirect_307(request: Request, exc):
    return RedirectResponse(
        url=exc.headers.get("Location", "/login"), status_code=307
    )
