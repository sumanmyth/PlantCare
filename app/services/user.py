from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.user import UserCreate
from passlib.hash import bcrypt

def create_user(db: Session, user: UserCreate):
    hashed_password = bcrypt.hash(user.password)  
    db_user = User(
        name=user.name,
        email=user.email,
        country_id=user.country_id,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_users(db: Session):
    return db.query(User).all()
