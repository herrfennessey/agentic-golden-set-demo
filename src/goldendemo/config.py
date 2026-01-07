"""
Configuration settings for the agentic golden set demo.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"
    wands_dir: Path = data_dir / "wands"
    golden_sets_dir: Path = data_dir / "golden_sets"

    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_collection_name: str = "WandsProduct"

    # OpenAI
    openai_api_key: str = ""  # typically starts with "sk-"
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Agent settings
    agent_max_iterations: int = 20
    agent_max_products: int = 500


settings = Settings()
