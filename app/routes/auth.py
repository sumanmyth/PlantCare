from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.schemas.user import UserCreate, UserResponse, Token
from app.services.auth import register_user,UserLogin,login_user, forgot_password, reset_password,get_google_auth_url, handle_google_callback
from fastapi.responses import JSONResponse

router = APIRouter()

# Register new user
@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = register_user(db, user)
    return db_user

# Login user
@router.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    return login_user(db, user)

@router.post("/forgot-password")
async def forgot_password_request(email: str, db: Session = Depends(get_db)):
    return await forgot_password(db, email)

@router.post("/reset-password")
def reset_password_request(token: str, new_password: str, db: Session = Depends(get_db)):
    return reset_password(db, token, new_password)


@router.get("/auth/login")
async def login_with_google():
    google_auth_url = get_google_auth_url()
    return JSONResponse(content={"redirect_url": google_auth_url})


# Route to handle Google OAuth2 callback
@router.get("/auth/callback")
async def google_auth_callback(code: str, db: Session = Depends(get_db)):
    try:
        user = await handle_google_callback(code, db)
        return {"user": user.email, "name": user.name}
    except HTTPException as e:
        raise e