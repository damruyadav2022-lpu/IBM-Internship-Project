"""
Voice Assistant module for KrishiRakshak.
Provides Speech-to-Text (STT) via Groq Whisper and
Text-to-Speech (TTS) via gTTS supporting Hindi, Punjabi, and English.
"""
import io
import logging
import os
import re
from typing import Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS

logger = logging.getLogger(__name__)

# Ensure env variables are loaded
load_dotenv()

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Punjabi": "pa",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Urdu": "ur",
}

LANGUAGE_CODE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "pa": "Punjabi",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ur": "Urdu",
}


def detect_language(text: str) -> Tuple[str, str]:
    """
    Auto-detect input language using Unicode script ranges and agricultural keywords.
    Returns (Language_Name, ISO_639_1_Code), e.g. ('Punjabi', 'pa'), ('Telugu', 'te').
    """
    if not text or not text.strip():
        return "English", "en"

    # 1. Unicode Script Checks (100% precision for native Indian scripts)
    if re.search(r'[\u0A00-\u0A7F]', text):  # Gurmukhi
        return "Punjabi", "pa"
    if re.search(r'[\u0980-\u09FF]', text):  # Bengali / Assamese
        return "Bengali", "bn"
    if re.search(r'[\u0C00-\u0C7F]', text):  # Telugu
        return "Telugu", "te"
    if re.search(r'[\u0B80-\u0BFF]', text):  # Tamil
        return "Tamil", "ta"
    if re.search(r'[\u0A80-\u0AFF]', text):  # Gujarati
        return "Gujarati", "gu"
    if re.search(r'[\u0C80-\u0CFF]', text):  # Kannada
        return "Kannada", "kn"
    if re.search(r'[\u0D00-\u0D7F]', text):  # Malayalam
        return "Malayalam", "ml"
    if re.search(r'[\u0600-\u06FF]', text):  # Perso-Arabic / Urdu
        return "Urdu", "ur"

    # Devanagari Script (Hindi vs Marathi distinction)
    if re.search(r'[\u0900-\u097F]', text):
        marathi_distinctive = ["आहे", "नाही", "पिक", "शेतकरी", "कसे", "करावे", "खत", "माहितੀ", "करा"]
        if any(w in text for w in marathi_distinctive):
            return "Marathi", "mr"
        return "Hindi", "hi"

    # 2. Transliterated Romanized Input (Hinglish / Punglish / Tanglish)
    t_lower = text.lower()
    hinglish_kw = [
        "kisan", "fasal", "khad", "tamatar", "gehu", "gehun", "kheti", "mandi",
        "bhav", "kaise", "batao", "dhan", "chawal", "rog", "bimari", "keet",
        "paani", "kitna", "kab", "kare", "karna", "upchar", "dawakhana"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', t_lower) for w in hinglish_kw):
        return "Hindi", "hi"

    punglish_kw = [
        "kanak", "jhone", "j 레이", "khet", "kisaan", "pind", "pani", "kadon",
        "kivein", "spray", "khungi", "keede", "sundi"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', t_lower) for w in punglish_kw):
        return "Punjabi", "pa"

    return "English", "en"


def _get_whisper_client() -> OpenAI:
    """Return OpenAI client configured for Groq Whisper."""
    load_dotenv(override=False)
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip()

    # Check Streamlit environment if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "session_state") and st.session_state.get("user_groq_api_key"):
            api_key = st.session_state.user_groq_api_key.strip()
        elif hasattr(st, "secrets"):
            if "LLM_API_KEY" in st.secrets:
                api_key = str(st.secrets["LLM_API_KEY"]).strip()
            elif "GROQ_API_KEY" in st.secrets:
                api_key = str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass

    return OpenAI(api_key=api_key, base_url=base_url)


def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "farmer_voice.wav",
    language: Optional[str] = None,
) -> str:
    """
    Transcribe audio bytes using Groq Whisper-large-v3.
    Supports all major Indian and international languages or auto-detection.
    """
    try:
        # Check Streamlit state for user entered API key
        try:
            import streamlit as st
            if hasattr(st, "session_state") and st.session_state.get("user_groq_api_key"):
                os.environ["LLM_API_KEY"] = st.session_state.user_groq_api_key.strip()
                os.environ["LLM_BASE_URL"] = "https://api.groq.com/openai/v1"
        except Exception:
            pass

        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key or "placeholder" in api_key.lower():
            raise RuntimeError("Groq API Key is not configured. Please enter your Groq API key (starts with 'gsk_') in the left sidebar under '🔑 AI API Key' or in Streamlit Cloud Secrets.")

        client = _get_whisper_client()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        # Map language string to ISO-639-1 code
        lang_code = None
        if language and language.lower() not in ("auto", "auto-detect", "none"):
            l_lower = language.lower()
            for name, code in SUPPORTED_LANGUAGES.items():
                if name.lower() in l_lower or code == l_lower:
                    lang_code = code
                    break

        kwargs = {
            "model": "whisper-large-v3",
            "file": audio_file,
        }
        if lang_code:
            kwargs["language"] = lang_code

        logger.info(f"Transcribing audio with Whisper (language={lang_code or 'auto-detect'})...")
        transcription = client.audio.transcriptions.create(**kwargs)
        return transcription.text.strip()
    except Exception as e:
        err_str = str(e)
        logger.error(f"Whisper transcription failed: {err_str}")
        if "401" in err_str or "invalid_api_key" in err_str.lower() or "expired" in err_str.lower():
            raise RuntimeError("Groq API Key is expired or invalid (HTTP 401). Please update your key (starts with 'gsk_') in the left sidebar or at https://console.groq.com/keys.")
        raise RuntimeError(f"Speech transcription failed: {err_str}")


def clean_text_for_speech(text: str) -> str:
    """
    Strip markdown formatting, tables, URLs, and symbols so TTS reads smoothly.
    """
    if not text:
        return ""
    # Remove markdown links [text](url) -> text
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove images
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned)
    # Remove markdown headers #, ##
    cleaned = re.sub(r'#+\s*', '', cleaned)
    # Remove bold/italic **text**, *text*, _text_
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
    cleaned = re.sub(r'__([^_]+)__', r'\1', cleaned)
    # Remove markdown table lines | col | col |
    cleaned = re.sub(r'\|[^\n]+\|', '', cleaned)
    cleaned = re.sub(r'[-:]{3,}', '', cleaned)
    # Remove emojis that might be pronounced awkwardly
    cleaned = re.sub(r'[`~>•📌🌾🔬🌤️💰🚨📞🌱🐛🧪💸✨🗣️🔊]', '', cleaned)
    # Remove multiple spaces/newlines
    cleaned = re.sub(r'\n{2,}', '. ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def generate_speech(
    text: str,
    language: str = "auto",
    max_chars: int = 1200,
) -> bytes:
    """
    Generate spoken MP3 audio from text using gTTS across all Indian languages.
    Auto-detects language if set to 'auto'.
    """
    clean_text = clean_text_for_speech(text)
    if not clean_text:
        clean_text = "No advice available."

    # Trim to avoid excessively long audio generation
    if len(clean_text) > max_chars:
        clean_text = clean_text[:max_chars].rsplit('.', 1)[0] + '.'

    # Language resolution
    lang_code = "en"
    if language and language.lower() not in ("auto", "auto-detect", "none"):
        l_lower = language.lower()
        for name, code in SUPPORTED_LANGUAGES.items():
            if name.lower() in l_lower or code == l_lower:
                lang_code = code
                break
    else:
        # Auto-detect language from text
        _, detected_code = detect_language(clean_text)
        lang_code = detected_code

    try:
        logger.info(f"Generating TTS audio (lang={lang_code}, length={len(clean_text)} chars)...")
        tts = gTTS(text=clean_text, lang=lang_code, slow=False)
        out_buf = io.BytesIO()
        tts.write_to_fp(out_buf)
        out_buf.seek(0)
        return out_buf.getvalue()
    except Exception as e:
        logger.error(f"TTS generation error for {lang_code}: {e}")
        # Fallback to English or Hindi
        fallback_code = "hi" if lang_code in ("pa", "mr", "gu") else "en"
        try:
            tts = gTTS(text=clean_text, lang=fallback_code, slow=False)
            out_buf = io.BytesIO()
            tts.write_to_fp(out_buf)
            out_buf.seek(0)
            return out_buf.getvalue()
        except Exception:
            return b""


