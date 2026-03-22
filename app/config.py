from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    # PostgreSQL via env var → auto switches from SQLite
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/rta.db"

    class Config:
        env_file = ".env"

settings = Settings()
