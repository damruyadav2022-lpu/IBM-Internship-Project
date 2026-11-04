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


def _get_whisper_client() -> OpenAI:
    """Return OpenAI client configured for Groq."""
    load_dotenv(override=True)
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "farmer_voice.wav",
    language: Optional[str] = None,
) -> str:
    """
    Transcribe audio bytes using Groq's whisper-large-v3.
    Supports Hindi ('hi'), Punjabi ('pa'), English ('en'), or auto-detection.
    """
    try:
        api_key = os.getenv("LLM_API_KEY", "")
        if not api_key or "placeholder" in api_key.lower():
            raise RuntimeError("LLM_API_KEY is not configured. Please add your Groq API key in Streamlit Cloud Secrets (or .env).")

        client = _get_whisper_client()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        # Map language string to ISO-639-1 code
        lang_code = None
        if language:
            l_lower = language.lower()
            if "hindi" in l_lower or l_lower == "hi":
                lang_code = "hi"
            elif "punjabi" in l_lower or l_lower == "pa":
                lang_code = "pa"
            elif "english" in l_lower or l_lower == "en":
                lang_code = "en"

        kwargs = {
            "model": "whisper-large-v3",
            "file": audio_file,
        }
        if lang_code:
            kwargs["language"] = lang_code

        logger.info(f"Transcribing audio with Whisper (language={lang_code})...")
        transcription = client.audio.transcriptions.create(**kwargs)
        return transcription.text.strip()
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise RuntimeError(f"Speech transcription failed: {str(e)}")


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
    # Keep punctuation and alphanumeric characters
    cleaned = re.sub(r'[`~>•📌🌾🔬🌤️💰🚨📞🌱🐛]', '', cleaned)
    # Remove multiple spaces/newlines
    cleaned = re.sub(r'\n{2,}', '. ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def generate_speech(
    text: str,
    language: str = "English",
    max_chars: int = 1200,
) -> bytes:
    """
    Generate spoken MP3 audio from text using gTTS.
    Supports Hindi ('hi'), Punjabi ('pa'), and English ('en').
    """
    clean_text = clean_text_for_speech(text)
    if not clean_text:
        clean_text = "No advice available."

    # Trim to avoid excessively long audio generation
    if len(clean_text) > max_chars:
        clean_text = clean_text[:max_chars].rsplit('.', 1)[0] + '.'

    # Language mapping
    l_lower = language.lower()
    if "hindi" in l_lower or l_lower == "hi":
        lang_code = "hi"
    elif "punjabi" in l_lower or l_lower == "pa":
        lang_code = "pa"
    else:
        lang_code = "en"

    try:
        logger.info(f"Generating TTS audio (lang={lang_code}, length={len(clean_text)} chars)...")
        tts = gTTS(text=clean_text, lang=lang_code, slow=False)
        out_buf = io.BytesIO()
        tts.write_to_fp(out_buf)
        out_buf.seek(0)
        return out_buf.getvalue()
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        # Fallback to English if regional language failed
        if lang_code != "en":
            try:
                tts = gTTS(text=clean_text, lang="en", slow=False)
                out_buf = io.BytesIO()
                tts.write_to_fp(out_buf)
                out_buf.seek(0)
                return out_buf.getvalue()
            except Exception:
                pass
        raise RuntimeError(f"Text-to-speech generation failed: {str(e)}")

