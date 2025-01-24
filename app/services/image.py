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

# Allowed image file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Function to check if the uploaded file is an image
def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

# Synchronous function to save the image info into the DB
def upload_image_to_db(db: Session, filename: str):
    image = Image(name=filename)
    db.add(image)
    db.commit()
    db.refresh(image)
    return image

# Load the model
model = tf.keras.models.load_model('app/plant_disease.h5')

# Read the CSV with labels and other information
df = pd.read_csv('app/p4.csv')

# Preprocess image as per model's requirement
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
    
    # Check if the file extension is valid
    if not is_image_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .jpg, .jpeg, .png, and .bmp files are allowed."
        )

    # Create upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Define file path and save the image
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)  # Save file to disk

    # Save image information in the database
    uploaded_image = upload_image_to_db(db, file.filename)  # This is a blocking operation, hence no 'await'
    
    # Preprocess the image and make a prediction
    processed_image = preprocess_image(file_path)
    prediction = model.predict(processed_image)

    # Get the top 3 predictions
    top3_indices = np.argsort(prediction[0])[-3:][::-1]
    top3_class_names = [df.iloc[i]['Label'] for i in top3_indices]
    top3_scores = prediction[0][top3_indices]
    top3_percentages = top3_scores / np.sum(top3_scores) * 100

    # Prepare the response with prediction results
    response = {}
    for i in range(3):
        index = top3_indices[i]
        treatment = df.iloc[index]['Treatment']
        if pd.isna(treatment):
            treatment = "No treatment needed"

        response[f"prediction_{i+1}"] = {
            "class_name": top3_class_names[i],
            "confidence": f"{top3_percentages[i]:.2f}%",
            "example_picture": file_path,
            "description": df.iloc[index]['Description'],
            "prevention": df.iloc[index]['Prevention'],
            "treatment": treatment
        }

    # Optionally, you can delete the saved image after processing
    # os.remove(file_path)

    # Return the JSON response with the prediction details
    return JSONResponse(content=response)
