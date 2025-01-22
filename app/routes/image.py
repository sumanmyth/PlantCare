from fastapi import APIRouter, Depends, UploadFile,File
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.models.image import Image
from app.schemas.image import ImageResponse,ImageCreate
from app.services.image import upload_image

image_router = APIRouter()

# @image_router.post("/upload-image", response_model=str)
# async def upload_image(file: UploadFile = File(...)):
#     try:
#         # Create a directory for storing images if it doesn't exist
#         if not os.path.exists("uploads"):
#             os.makedirs("uploads")
        
#         # Save the uploaded file to the "uploads" directory
#         file_path = f"uploads/{file.filename}"
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         return JSONResponse(status_code=200, content={"message": "Image uploaded successfully"})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



@image_router.post("/upload-image", response_model=ImageResponse)
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    uploaded_image = await upload_image(file, db)
    return uploaded_image
