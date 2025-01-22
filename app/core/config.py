from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    smtp_username: str
    smtp_password: str
    email_from: str
    smtp_server: str
    smtp_port: int  # This ensures the port is treated as an integer

    class Config:
        env_file = ".env"
        extra = "allow"

# Create the settings instance
settings = Settings()
