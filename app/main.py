from fastapi import FastAPI, File, UploadFile, HTTPException
from app.routes.image import image_router # Import the image router
from app.core.db import Base, engine
from app.routes import user
from app.routes import auth  # Import user router from routes folder
from app.models.user import User
from app.schemas.user import UserCreate 
from app.services.user import create_user 
import os 
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError


app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="add any string...")
load_dotenv()

oauth = OAuth()
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    client_kwargs={
        'scope': 'email openid profile',
        'redirect_url': 'http://localhost:8000/auth'
    }
)


# Ensure you're including the user router correctly
app.include_router(user.router, prefix="/api/users", tags=["users"])

app.include_router(auth.router, prefix="/api", tags=["auth"])

# Include the image router 
app.include_router(image_router, prefix="/api", tags=["images"])

# Create all tables in the database
Base.metadata.create_all(bind=engine)

@app.get("/api")
def read_root():
    return {"message": "Welcome to the FastAPI Backend"}

# Add the endpoint for image uploads
# @app.post("/api/images/upload", response_model=str)
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

# @app.get("/")
# def index(request: Request):
#     user = request.session.get('user')
#     if user:
#         return RedirectResponse('welcome')


