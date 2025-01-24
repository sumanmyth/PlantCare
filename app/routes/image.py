from fastapi import APIRouter, Depends, UploadFile,File
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.models.image import Image
from app.schemas.image import ImageResponse,ImageCreate
from app.services.image import upload_image

image_router = APIRouter()

@image_router.post("/upload-image", response_model=ImageResponse)
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    uploaded_image = await upload_image(file, db)
    return uploaded_image
