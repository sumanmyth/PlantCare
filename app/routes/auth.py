from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.schemas.user import UserCreate, UserResponse, Token
from app.services.auth import (
    register_user,
    UserLogin,
    login_user,
    forgot_password,
    reset_password,
    get_google_auth_url,
    handle_google_callback,
    logout_user
)
from fastapi.responses import JSONResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Register new user
@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    try:
        db_user = register_user(db, user)
        return db_user
    except HTTPException as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=400, detail="Registration failed")

# Login user
@router.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    try:
        return login_user(db, user)
    except HTTPException as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
 # Logout user 
@router.post("/logout")
def logout(token: str, db: Session = Depends(get_db)):
    try:
        return logout_user(db, token)
    except HTTPException as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(status_code=400, detail="Logout failed")

# Forgot password
@router.post("/forgot-password")
async def forgot_password_request(email: str, db: Session = Depends(get_db)):
    try:
        return await forgot_password(db, email)
    except HTTPException as e:
        logger.error(f"Forgot password request failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Email not found")

# Reset password
@router.post("/reset-password")
def reset_password_request(token: str, new_password: str, db: Session = Depends(get_db)):
    try:
        return reset_password(db, token, new_password)
    except HTTPException as e:
        logger.error(f"Password reset failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid or expired token")

# Google OAuth Login
@router.get("/auth/login")
async def login_with_google():
    try:
        google_auth_url = get_google_auth_url()
        return JSONResponse(content={"redirect_url": google_auth_url})
    except HTTPException as e:
        logger.error(f"Google OAuth login failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Google authentication failed")

# Google OAuth Callback
@router.get("/auth/callback")
async def google_auth_callback(code: str, db: Session = Depends(get_db)):
    try:
        user = await handle_google_callback(code, db)
        return {"user": user.email, "name": user.name}
    except HTTPException as e:
        logger.error(f"Google auth callback error: {str(e)}")
        raise HTTPException(status_code=400, detail="Google authentication failed")
    

