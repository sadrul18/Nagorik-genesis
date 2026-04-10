"""
Configuration module for নাগরিক-GENESIS (NAGORIK-GENESIS).
Centralize configuration and environment variables for Bangladesh policy simulator.
"""
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Settings:
    """Application settings and configuration."""

    # LLM Backend: "ollama" (local) or "gemini" (cloud)
    llm_backend: str = "ollama"
    ollama_model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"

    # API Keys (only needed when llm_backend="gemini")
    gemini_api_key: str = ""
    backup_api_keys: Optional[List[str]] = None
    summary_api_key: Optional[str] = None

    # Default simulation parameters
    default_population_size: int = 1000
    default_steps: int = 5
    max_population_size: int = 50000

    # LLM sampling configuration
    llm_sample_size: int = 300  # Max citizens to sample with LLM per step

    # Neural network configuration — larger for 40-dim feature space
    nn_hidden_layers: Tuple[int, ...] = (128, 64, 32)
    nn_max_iter: int = 500
    nn_min_training_samples: int = 500

    # Random seed (optional)
    random_seed: Optional[int] = None

    # Income distribution defaults — Bangladesh reality (~55% low, ~35% middle, ~10% high)
    low_income_share: float = 0.55
    middle_income_share: float = 0.35
    high_income_share: float = 0.10


def get_settings() -> Settings:
    """
    Get application settings from environment variables.

    Returns:
        Settings object with configuration values.

    Raises:
        ValueError: If backend=gemini but GEMINI_API_KEY is missing.
    """
    llm_backend = os.getenv("LLM_BACKEND", "ollama").strip().lower()
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()
    ollama_host  = os.getenv("OLLAMA_HOST",  "http://localhost:11434").strip()

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if llm_backend == "gemini" and not gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY is required when LLM_BACKEND=gemini. "
            "Set it in .env or switch to LLM_BACKEND=ollama."
        )

    backup_keys_str = os.getenv("GEMINI_BACKUP_KEYS", "")
    backup_keys = [k.strip() for k in backup_keys_str.split(",") if k.strip()]
    summary_api_key = os.getenv("GEMINI_SUMMARY_API_KEY", "").strip() or None

    return Settings(
        llm_backend=llm_backend,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        gemini_api_key=gemini_api_key,
        backup_api_keys=backup_keys if backup_keys else None,
        summary_api_key=summary_api_key,
        random_seed=int(os.getenv("RANDOM_SEED", 0)) or None
    )
