from app.schemas.image import ImageResponse
from app.schemas.prediction import DiseasePredictionResponse
from app.core.db import get_db
from sqlalchemy.orm import Session
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
import os
import shutil
import numpy as np
from PIL import Image as PILImage
import io
from app.models.image import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse
import cv2
import pandas as pd

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def upload_image_to_db(db: Session, filename: str):
    image = Image(name=filename)
    db.add(image)
    db.commit()
    db.refresh(image)
    return image

model = tf.keras.models.load_model('app/plant_disease.h5')
df = pd.read_csv('app/p4.csv')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to read image. Ensure the file is a valid image format.")
    
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    upload_dir = "uploads"
    
    if not is_image_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .jpg, .jpeg, .png, and .bmp files are allowed."
        )

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    uploaded_image = upload_image_to_db(db, file.filename)
    processed_image = preprocess_image(file_path)
    prediction = model.predict(processed_image)
    
    top_index = np.argmax(prediction[0])
    confidence = prediction[0][top_index] * 100  # Convert to percentage
    class_name = df.iloc[top_index]['Label']
    treatment = df.iloc[top_index]['Treatment']
    if pd.isna(treatment):
        treatment = "No treatment needed"

    if confidence < 90:
        return JSONResponse(content={"message": "The uploaded image does not appear to be a plant."})
    
    response = {
        "class_name": class_name,
        "confidence": f"{confidence:.2f}%",
        "example_picture": file_path,
        "description": df.iloc[top_index]['Description'],
        "prevention": df.iloc[top_index]['Prevention'],
        "treatment": treatment
    }
    
    return JSONResponse(content=response)
