"""
app/routers/auth.py
Authentication - uses werkzeug password hashing (no length limits).
"""
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.user import User
from app.schemas.user import TokenData

logger = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])
templates = Jinja2Templates(directory="app/templates")


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return check_password_hash(hashed, plain)


def create_access_token(email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": email, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> TokenData | None:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return TokenData(email=email)
    except JWTError:
        return None


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    from fastapi import HTTPException
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=307, headers={"Location": f"/login?next={request.url.path}"})
    token_data = decode_token(token)
    if not token_data:
        raise HTTPException(status_code=307, headers={"Location": "/login"})
    result = await db.execute(select(User).where(User.email == token_data.email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=307, headers={"Location": "/login"})
    return user


async def get_current_user_or_none(request: Request, db: AsyncSession = Depends(get_db)) -> User | None:
    try:
        return await get_current_user(request, db)
    except Exception:
        return None


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/dashboard", error: str = ""):
    return templates.TemplateResponse("login.html", {"request": request, "next": next, "error": error, "tab": "login"})


@router.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form(default="/dashboard"),
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await db.execute(select(User).where(User.email == email.lower().strip()))
        user = result.scalar_one_or_none()
        if not user or not verify_password(password, user.hashed_password):
            return templates.TemplateResponse("login.html",
                {"request": request, "next": next, "error": "Invalid email or password.", "tab": "login"},
                status_code=400)
        token = create_access_token(user.email)
        response = RedirectResponse(url=next, status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(key="access_token", value=token, httponly=True,
                            max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60, samesite="lax")
        return response
    except Exception as e:
        logger.exception("Login error: %s", e)
        return templates.TemplateResponse("login.html",
            {"request": request, "next": next, "error": f"Login failed: {str(e)}", "tab": "login"},
            status_code=500)


@router.post("/register")
async def register_submit(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    try:
        errors = []
        if len(full_name.strip()) < 2:
            errors.append("Full name must be at least 2 characters.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if password != confirm_password:
            errors.append("Passwords do not match.")
        if errors:
            return templates.TemplateResponse("login.html",
                {"request": request, "next": "/dashboard", "error": " ".join(errors), "tab": "register"},
                status_code=400)

        existing = await db.execute(select(User).where(User.email == email.lower().strip()))
        if existing.scalar_one_or_none():
            return templates.TemplateResponse("login.html",
                {"request": request, "next": "/dashboard", "error": "Email already registered.", "tab": "register"},
                status_code=400)

        user = User(full_name=full_name.strip(), email=email.lower().strip(), hashed_password=hash_password(password))
        db.add(user)
        await db.commit()
        await db.refresh(user)

        token = create_access_token(user.email)
        response = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(key="access_token", value=token, httponly=True,
                            max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60, samesite="lax")
        return response
    except Exception as e:
        logger.exception("Register error: %s", e)
        return templates.TemplateResponse("login.html",
            {"request": request, "next": "/dashboard", "error": f"Registration failed: {str(e)}", "tab": "register"},
            status_code=500)


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("access_token")
    return response
