"""
Utility helpers to load prompts and instantiate the chat LLM.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml
from langchain_anthropic import ChatAnthropic

CONFIG_DIR = Path(__file__).resolve().parent


def _read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache
def load_private_config() -> Dict[str, str]:
    """
    Load secrets from config/private_config.yaml or environment variables.

    Returns:
        Dict with configuration keys.
    """
    candidate = CONFIG_DIR / "private_config.yaml"
    if candidate.exists():
        return _read_yaml(candidate)
    return {}


@lru_cache
def load_prompt_template(name: str) -> str:
    prompts = _read_yaml(CONFIG_DIR / "promts.yaml")
    if name not in prompts:
        raise KeyError(f"Prompt template '{name}' not found.")
    return prompts[name]


@lru_cache
def load_llm(model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.1) -> ChatAnthropic:
    """
    Instantiate the Anthropic chat model using config or environment secrets.
    """
    secrets = load_private_config()
    api_key = secrets.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key missing. Set 'anthropic_api_key' in config/private_config.yaml "
            "or export the ANTHROPIC_API_KEY environment variable."
        )
    return ChatAnthropic(model=model, anthropic_api_key=api_key, temperature=temperature)


__all__ = ["load_llm", "load_prompt_template", "load_private_config"]
