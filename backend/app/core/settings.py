from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    APP_NAME: str = "Multi-Modal AI"
    APP_ENV: str = Field(default="development")
    DEBUG: bool = Field(default=True)

    API_PREFIX: str = "/api"

    OPENAI_API_KEY: str = Field(default="")
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    VECTOR_DB_PATH: str = Field(default="storage/chroma_db")
    UPLOAD_DIR: str = Field(default="storage/uploads")
    LOG_LEVEL: str = Field(default="INFO")
    CACHE_EXPIRE_SECONDS: int = Field(default=3600)

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()