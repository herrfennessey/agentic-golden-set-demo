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

    # OpenAI
    openai_api_key: str = ""  # typically starts with "sk-"
    openai_embedding_model: str = "text-embedding-3-small"  # Used by Weaviate vectorizer

    # Agent settings
    agent_max_iterations: int = 20
    agent_model: str = "gpt-5-nano"
    agent_reasoning_effort: str = "medium"  # low, medium, high
    agent_reasoning_summary: bool = True

    # Judgment subagent settings (isolated context for product evaluation)
    judge_model: str = "gpt-5-nano"
    judge_reasoning_effort: str = "low"  # low for structured tasks - high wastes tokens on reasoning
    judge_chunk_size: int = 25  # Products per judgment batch
    judge_max_workers: int = 5  # Parallel judgment workers
    judge_max_retries: int = 2  # Retry attempts on failure
    # Note: No max_output_tokens - letting it truncate wastes money (pay for truncated + retry)

    # Validation subagent settings (reviews judgments before save)
    validate_model: str = "gpt-5-nano"
    validate_reasoning_effort: str = "medium"  # Higher than judge - needs to catch errors
    validate_chunk_size: int = 20  # Smaller chunks for more careful review
    validate_max_workers: int = 3  # Fewer parallel workers for reliability
    validate_max_retries: int = 2

    # Category browsing limits
    browse_product_limit: int = 2000  # Max products to fetch per category

    # Golden set composition thresholds
    min_exact_judgments: int = 5  # Minimum Exact (2) judgments required
    min_total_judgments: int = 50  # Minimum total judgments required


settings = Settings()
