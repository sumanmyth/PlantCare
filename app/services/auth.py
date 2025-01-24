from datetime import datetime, timedelta
from typing import Union
from passlib.hash import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin
from app.core.db import get_db
from fastapi import HTTPException
from app.core.mail import send_email
import os
import httpx
from fastapi import HTTPException

# Secret key for JWT encoding and decoding
SECRET_KEY = "SUMAN977"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: timedelta = None):
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def register_user(db: Session, user: UserCreate):
    hashed_password = bcrypt.hash(user.password)
    db_user = User(
        name=user.name,
        email=user.email,
        country_id=user.country_id,
        password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if not user or not bcrypt.verify(password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user

def login_user(db: Session, user: UserLogin):
    authenticated_user = authenticate_user(db, user.email, user.password)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": authenticated_user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

def create_reset_token(data: dict, expires_delta: timedelta = None):
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def forgot_password(db: Session, email: str):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    reset_token = create_reset_token(data={"sub": email})
    
    # Send reset email with token
    reset_url = f"http://localhost:8000/reset-password?token={reset_token}"
    email_body = f"Click the link to reset your password: {reset_url}"
    await send_email(subject="Password Reset", recipient_email=[email], body=email_body)
    
    return {"message": "Password reset email sent"}


def reset_password(db: Session, token: str, new_password: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        user.hashed_password = bcrypt.hash(new_password)
        db.commit()
        db.refresh(user)
        return {"message": "Password updated successfully"}
    
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# Function to start the Google OAuth login flow
def get_google_auth_url():
    google_auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?response_type=code"
        f"&client_id={GOOGLE_CLIENT_ID}&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&scope=openid%20email&state=random_state_string"
    )
    return google_auth_url


# Function to handle the Google OAuth callback and get the user info
async def handle_google_callback(code: str, db: Session):
    token_url = "https://oauth2.googleapis.com/token"
    
    # Prepare data for exchanging code for tokens
    token_data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=token_data)

    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Error fetching token from Google.")
    
    token_info = response.json()
    id_token = token_info.get("id_token")
    if not id_token:
        raise HTTPException(status_code=400, detail="ID token not found.")

    # Decode the ID token to extract user info
    try:
        decoded_token = jwt.decode(id_token, options={"verify_signature": False})
        email = decoded_token.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Email not found in the ID token.")
        
        # Check if user exists in the DB
        user = db.query(User).filter(User.email == email).first()
        if not user:
            # Create a new user if not found
            user = User(email=email, name=decoded_token.get("name"))
            db.add(user)
            db.commit()
            db.refresh(user)

        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=400, detail="Invalid token")

def logout_user(db: Session, token: str):
    # This is where you'd add logic to handle token invalidation or blacklisting.
    return {"message": "Logged out successfully"}