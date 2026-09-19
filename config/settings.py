# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env",env_file_encoding="utf-8",extra="ignore",)

    # DB
    pg_connection_string: str = Field(...,alias="PG_CONNECTION_STRING")
    pg_collection_name : str = Field(...,alias="PG_COLLECTION_NAME")

    # LLM/Embedding
    google_api_key: str = Field(...,alias="GOOGLE_API_KEY")
    mistral_api_key: str = Field(...,alias="MISTRAL_API_KEY")


    # Retrieval tuning
    top_k: int = 5
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

    # App
    debug: bool = False

settings = Settings()