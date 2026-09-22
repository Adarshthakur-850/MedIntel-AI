import os
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "MedIntel AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("JWT_SECRET", "MEDINTEL_SECRET_KEY_PRODUCTION_REPLACE_IN_ENV_6781263")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./medintel.db")

settings = Settings()
