from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "medical-rag"
    log_level: str = "INFO"
    allowed_origins: list[str] = []

    openai_api_key: str | None = None
    vector_store_path: str = "data/index/vector_store.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
