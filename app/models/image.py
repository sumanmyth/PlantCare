from sqlalchemy import Column, Integer, String
from app.core.db import Base

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)