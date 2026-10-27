from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "books"
    pinecone_namespace: str = "books-v1"
    pinecone_embed_model: str = "llama-text-embed-v2"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings. Restart the server if you change .env."""
    return Settings()
