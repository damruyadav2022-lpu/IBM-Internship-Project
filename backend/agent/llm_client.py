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


def _resolve_config():
    """Resolve LLM credentials across os.environ, Streamlit session_state, and Streamlit secrets."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None

    # Check Streamlit environment if running inside Streamlit
    try:
        import streamlit as st
        # 1. User entered key in sidebar UI (from explicit save or active input widget)
        if hasattr(st, "session_state"):
            if st.session_state.get("user_groq_api_key"):
                api_key = st.session_state.user_groq_api_key.strip()
            elif st.session_state.get("input_groq_key_sidebar"):
                api_key = st.session_state.input_groq_key_sidebar.strip()
            if api_key:
                os.environ["LLM_API_KEY"] = api_key

        # 2. Key stored in Streamlit Cloud Secrets
        if (not api_key or api_key == "sk-placeholder") and hasattr(st, "secrets"):
            if "LLM_API_KEY" in st.secrets:
                api_key = str(st.secrets["LLM_API_KEY"]).strip()
                os.environ["LLM_API_KEY"] = api_key
            elif "GROQ_API_KEY" in st.secrets:
                api_key = str(st.secrets["GROQ_API_KEY"]).strip()
                os.environ["LLM_API_KEY"] = api_key
            elif "GROQ_KEY" in st.secrets:
                api_key = str(st.secrets["GROQ_KEY"]).strip()
                os.environ["LLM_API_KEY"] = api_key

            if not base_url and "LLM_BASE_URL" in st.secrets:
                base_url = str(st.secrets["LLM_BASE_URL"]).strip()
                os.environ["LLM_BASE_URL"] = base_url
            if "LLM_MODEL" in st.secrets:
                model = str(st.secrets["LLM_MODEL"]).strip()
                os.environ["LLM_MODEL"] = model
    except Exception:
        pass

    # Ensure Groq endpoint is used by default or whenever key is a Groq key (starts with gsk_)
    if not base_url or api_key.startswith("gsk_") or "120b" in model.lower() or "llama" in model.lower():
        if not base_url or "openai.com/v1" in base_url and not base_url.startswith("https://api.groq"):
            base_url = "https://api.groq.com/openai/v1"
            os.environ["LLM_BASE_URL"] = base_url

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    return provider, api_key, model, base_url, temperature, max_tokens


def get_llm() -> BaseLanguageModel:
    """
    Instantiate and return an LLM based on environment configuration.
    Supports any OpenAI-compatible endpoint (Groq, OpenAI, Ollama, HuggingFace TGI).
    """
    # Re-read .env if present
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE, override=False)

    provider, api_key, model, base_url, temperature, max_tokens = _resolve_config()

    if not api_key or api_key == "sk-placeholder":
        raise ValueError("API_KEY_NOT_CONFIGURED")

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

    # Default: OpenAI or Groq via ChatOpenAI
    kwargs: dict = dict(
        model=model,
        openai_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if base_url:
        kwargs["openai_api_base"] = base_url

    return ChatOpenAI(**kwargs)
