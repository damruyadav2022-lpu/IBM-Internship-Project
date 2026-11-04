"""
LLM client — wraps OpenAI-compatible APIs (OpenAI, Groq, Ollama, HuggingFace TGI, etc.)
Configuration via environment variables in .env:
  LLM_PROVIDER   - "openai" | "ollama" | "huggingface" (default: "openai")
  LLM_API_KEY    - API key (use "ollama" for local Ollama)
  LLM_MODEL      - model name (e.g. "openai/gpt-oss-120b" or "qwen/qwen3.8-27b")
  LLM_BASE_URL   - custom base URL for OpenAI-compatible endpoints (e.g. "https://api.groq.com/openai/v1")
  LLM_TEMPERATURE - sampling temperature (default: 0.3)
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)

# Ensure .env is always loaded from project root
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE, override=True)


def get_llm() -> BaseLanguageModel:
    """
    Instantiate and return an LLM based on environment configuration.
    Supports any OpenAI-compatible endpoint (OpenAI, Groq, Ollama, HuggingFace TGI,
    LM Studio, Together AI, etc.) via the openai-compatible client.
    """
    # Re-read .env to catch any live updates without requiring full restart
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE, override=True)

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key = os.getenv("LLM_API_KEY", "sk-placeholder")
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    base_url = os.getenv("LLM_BASE_URL", None)
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    logger.info(f"Initializing LLM: provider={provider}, model={model}, base_url={base_url}")

    if provider == "ollama":
        ollama_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return ChatOpenAI(
            model=model or "llama2",
            openai_api_key="ollama",
            openai_api_base=ollama_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "huggingface":
        hf_url = base_url or os.getenv(
            "HF_ENDPOINT_URL",
            "https://api-inference.huggingface.co/v1",
        )
        hf_key = api_key or os.getenv("HF_API_KEY", "hf-placeholder")
        return ChatOpenAI(
            model=model or "mistralai/Mistral-7B-Instruct-v0.2",
            openai_api_key=hf_key,
            openai_api_base=hf_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Default: OpenAI or any generic OpenAI-compatible API (e.g. Groq)
    kwargs: dict = dict(
        model=model,
        openai_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if base_url:
        kwargs["openai_api_base"] = base_url

    return ChatOpenAI(**kwargs)
