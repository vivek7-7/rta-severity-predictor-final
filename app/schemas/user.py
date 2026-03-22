"""
app/schemas/user.py
Pydantic schemas for user registration, login, and JWT token handling.
"""

from pydantic import BaseModel, EmailStr, field_validator, ConfigDict
import re


class UserRegister(BaseModel):
    """Schema for new user registration."""

    full_name: str
    email: EmailStr
    password: str
    confirm_password: str

    @field_validator("full_name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Full name must be at least 2 characters.")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match.")
        return v


class UserLogin(BaseModel):
    """Schema for login form submission."""

    email: EmailStr
    password: str


class UserOut(BaseModel):
    """Public-safe user representation (no password)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: str


class Token(BaseModel):
    """JWT access token response."""

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Decoded JWT payload."""

    email: str | None = None
