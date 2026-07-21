from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
  
    db_url: str
    session_expire_seconds: int 

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()