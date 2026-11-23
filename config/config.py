from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "face-auth"
    db_url: str
    jwt_secret: str  # required: app refuses to start without it
    jwt_algorithm: str = "HS256"
    session_expire_seconds: int = 3600
    cookie_secure: bool = False  # True behind HTTPS
    cors_origins: list[str] = ["http://localhost:3000"]
    match_threshold: float = 0.80  # cosine distance; calibrate on your own data
    embedder_model_id: str = "minchul/cvlface_arcface_ir101_webface4m"
    log_file: str = "logs/app.jsonl"
    log_level: str = "INFO"


settings = Settings()
