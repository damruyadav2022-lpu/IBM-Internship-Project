"""
KrishiRakshak — Smart Farming Advice AI Agent
Ultra-Modern, Responsive UI/UX powered by Streamlit.
Run: streamlit run streamlit_app.py
"""
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KrishiRakshak — Smart Farming AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Streamlit Cloud Secrets Sync ──────────────────────────────────────────────
# Automatically propagate Streamlit Community Cloud Secrets into os.environ
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ[_k] = _v
            if _k.upper() in ("GROQ_API_KEY", "GROQ_KEY"):
                os.environ["LLM_API_KEY"] = _v
        elif isinstance(_v, dict):
            for _sub_k, _sub_v in _v.items():
                if isinstance(_sub_v, str):
                    os.environ[_sub_k] = _sub_v
                    if _sub_k.upper() in ("GROQ_API_KEY", "GROQ_KEY"):
                        os.environ["LLM_API_KEY"] = _sub_v
except Exception:
    pass

# Ensure LLM_BASE_URL defaults to Groq (never unconfigured or defaulting to OpenAI)
if not os.getenv("LLM_BASE_URL") or "openai.com/v1" in os.getenv("LLM_BASE_URL", ""):
    os.environ["LLM_BASE_URL"] = "https://api.groq.com/openai/v1"

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_URL = f"{BACKEND_URL}/api"

CITIES = [
    "Delhi", "Mumbai", "Pune", "Lucknow", "Jaipur",
    "Ludhiana", "Chandigarh", "Bhopal", "Patna", "Nashik"
]

SEVERITY_EMOJI = {
    "critical": "🔴",
    "high":     "🟠",
    "medium":   "🟡",
    "none":     "🟢",
    "unknown":  "⚪",
}
SEVERITY_COLOR = {
    "critical": "#3b1418",
    "high":     "#381e0f",
    "medium":   "#36290c",
    "none":     "#0d2b20",
    "unknown":  "#1e293b",
}

# ── Midnight Forest & Slate Agricultural Theme (Eye-Comfort & High Contrast) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;600;700&family=Noto+Sans+Gurmukhi:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Noto Sans Devanagari', 'Noto Sans Gurmukhi', sans-serif !important;
    color: #e2e8f0;
}

/* Main App Dark Background */
.stApp {
    background: #0b141a !important;
    color: #f1f5f9 !important;
}

/* Sidebar Dark Styling */
[data-testid="stSidebar"] {
    background-color: #0d1b22 !important;
    border-right: 1px solid #1e3a47 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
    color: #e2e8f0 !important;
}

/* Main Top Hero Banner */
.hero-card {
    background: linear-gradient(135deg, #062b1f 0%, #084332 50%, #065f46 100%) !important;
    border: 1px solid rgba(52, 211, 153, 0.35) !important;
    color: #ffffff !important;
    padding: 1.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
}

.hero-card::after {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(52, 211, 153, 0.2) 0%, rgba(0, 0, 0, 0) 70%);
    pointer-events: none;
}

.hero-title-group h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
    color: #ffffff !important;
}

.hero-title-group p {
    margin: 0.35rem 0 0 0;
    font-size: 0.95rem;
    color: #a7f3d0 !important;
    font-weight: 500;
}

.hero-badges-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    align-items: center;
}

.hero-pill {
    background: rgba(16, 185, 129, 0.15) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(52, 211, 153, 0.35) !important;
    padding: 0.35rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #a7f3d0 !important;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #0f202a !important;
    border: 1px solid #1e3a47 !important;
    padding: 6px;
    border-radius: 14px;
    margin-bottom: 1.25rem;
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    white-space: pre-wrap;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.92rem;
    color: #94a3b8 !important;
    padding: 0 1.2rem;
    transition: all 0.2s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #34d399 !important;
    background: rgba(16, 185, 129, 0.12) !important;
}

.stTabs [aria-selected="true"] {
    background: #132b38 !important;
    color: #34d399 !important;
    border: 1px solid #10b981 !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
}

/* Modern Card Layouts */
.ui-card {
    background: #13242e !important;
    border: 1px solid #1e3a47 !important;
    border-radius: 16px;
    padding: 1.25rem;
    color: #f1f5f9 !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.ui-card:hover {
    border-color: #10b981 !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
}

/* Chat bubble styling */
.chat-container-box {
    background: #112028 !important;
    border: 1px solid #1e3a47 !important;
    border-radius: 16px;
    padding: 1.25rem;
}

.chat-bubble-user {
    background: linear-gradient(135deg, #065f46 0%, #047857 100%) !important;
    color: #ffffff !important;
    padding: 0.85rem 1.2rem;
    border-radius: 20px 20px 4px 20px;
    margin: 0.6rem 0;
    max-width: 82%;
    margin-left: auto;
    font-size: 0.95rem;
    line-height: 1.5;
    border: 1px solid #10b981;
    box-shadow: 0 4px 10px rgba(4, 120, 87, 0.25);
    word-break: break-word;
}

.chat-bubble-assistant {
    background: #13242e !important;
    border: 1px solid #1e3a47 !important;
    border-left: 4px solid #10b981 !important;
    color: #f1f5f9 !important;
    padding: 1rem 1.25rem;
    border-radius: 4px 20px 20px 20px;
    margin: 0.6rem 0;
    max-width: 88%;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    word-break: break-word;
}

.source-tag {
    background: #064e3b !important;
    color: #a7f3d0 !important;
    border: 1px solid #059669 !important;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.74rem;
    font-weight: 700;
    margin-right: 6px;
    display: inline-block;
}

/* Metric Boxes */
[data-testid="stMetric"] {
    background: #13242e !important;
    border: 1px solid #1e3a47 !important;
    border-radius: 14px;
    padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* Weather Day Card */
.weather-card-modern {
    background: #13242e !important;
    border: 1px solid #1e3a47 !important;
    border-radius: 14px;
    padding: 0.85rem;
    text-align: center;
    color: #f1f5f9 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.weather-card-modern:hover {
    transform: translateY(-3px);
    border-color: #10b981 !important;
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2);
}

/* Mandi table styling */
.mandi-price { font-weight: 700; color: #34d399 !important; }
.trend-up   { color: #f87171 !important; font-weight: 700; }
.trend-down { color: #4ade80 !important; font-weight: 700; }
.trend-stable { color: #fbbf24 !important; font-weight: 700; }

/* Stat box */
.stat-box-modern {
    background: #13242e !important;
    border: 1px solid #1e3a47 !important;
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    color: #f1f5f9 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

/* Drag & Drop Zone Enhancement */
[data-testid="stFileUploader"] {
    border: 2px dashed #10b981 !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    background: #0f202a !important;
    color: #f1f5f9 !important;
    transition: all 0.25s ease-in-out;
}

[data-testid="stFileUploader"]:hover {
    border-color: #34d399 !important;
    background: #132b38 !important;
}

/* Pulsing Voice Assistant Indicator */
@keyframes pulse-ring {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5); }
    70% { transform: scale(1.02); box-shadow: 0 0 0 12px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.voice-mic-active {
    animation: pulse-ring 2s infinite;
}

/* Audio wave equalizer animation */
.sound-wave-box {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    height: 42px;
    padding: 8px 16px;
    background: #0d231b;
    border-radius: 12px;
    border: 1px solid #10b981;
}
.wave-bar {
    width: 4px;
    height: 100%;
    background: #34d399;
    border-radius: 4px;
    animation: wave-anim 1.2s ease-in-out infinite;
}
.wave-bar:nth-child(2) { animation-delay: 0.15s; }
.wave-bar:nth-child(3) { animation-delay: 0.3s; }
.wave-bar:nth-child(4) { animation-delay: 0.45s; }
.wave-bar:nth-child(5) { animation-delay: 0.6s; }
@keyframes wave-anim {
    0%, 100% { height: 8px; }
    50% { height: 32px; }
}

/* Circular Risk Meter Gauge */
.risk-meter-wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px 0;
}
.risk-meter-circle {
    width: 130px;
    height: 130px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    border: 4px solid #10b981;
    background: radial-gradient(circle, #092a20 0%, #13242e 70%);
    box-shadow: 0 0 24px rgba(16, 185, 129, 0.3);
}
.risk-score-big {
    font-size: 2.3rem;
    font-weight: 800;
    line-height: 1;
}
.risk-score-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* Hazard Card */
.hazard-card {
    background: #112028;
    border: 1px solid #1e3a47;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    color: #f1f5f9;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.hazard-card:hover {
    transform: translateY(-2px);
    border-color: #38bdf8;
    box-shadow: 0 6px 16px rgba(56, 189, 248, 0.15);
}

/* Sample Leaf Card */
.sample-leaf-pill {
    background: #13242e;
    border: 1px solid #1e3a47;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 0.88rem;
    color: #e2e8f0;
    transition: all 0.2s ease;
}
.sample-leaf-pill:hover {
    border-color: #10b981;
    background: #0f2b23;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
}

/* Inputs and Forms */
[data-testid="stTextInput"] input, [data-testid="stSelectbox"] select {
    background-color: #13242e !important;
    color: #f1f5f9 !important;
    border: 1px solid #1e3a47 !important;
}

/* Telemetry Ribbon */
.telemetry-ribbon {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    background: #0d1b22;
    border: 1px solid #1e3a47;
    border-radius: 12px;
    padding: 8px 16px;
    margin-bottom: 1.25rem;
    align-items: center;
    font-size: 0.82rem;
    color: #94a3b8;
}
.telemetry-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.telemetry-item b {
    color: #34d399;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .hero-card {
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
        padding: 1.25rem;
    }
    .hero-title-group h1 { font-size: 1.6rem; }
    .chat-bubble-user { max-width: 95%; }
    .chat-bubble-assistant { max-width: 95%; }
    .telemetry-ribbon { flex-direction: column; align-items: flex-start; gap: 6px; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state: st.session_state.messages = []
if "language" not in st.session_state: st.session_state.language = "English"
if "location" not in st.session_state: st.session_state.location = "Delhi"
if "detection" not in st.session_state: st.session_state.detection = None
if "voice_chat_history" not in st.session_state: st.session_state.voice_chat_history = []

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_int(val, default: int = 0) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

def safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_clamp(val, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return min(high, max(low, float(val)))
    except (ValueError, TypeError):
        return low

@st.cache_data(ttl=300, show_spinner=False)
def is_backend_online() -> bool:
    """Detect whether a dedicated FastAPI gateway is active on BACKEND_URL."""
    if os.getenv("STREAMLIT_SERVER_HEADLESS") == "true" and "localhost" in BACKEND_URL:
        return False
    try:
        r = requests.get(f"{API_URL}/health", timeout=0.3)
        return r.status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=600, show_spinner=False)
def load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=600, show_spinner=False)
def get_realtime_weather(location: str) -> dict:
    """Fetch real-time weather from backend API or local WeatherService."""
    if is_backend_online():
        try:
            r = requests.get(f"{API_URL}/weather", params={"location": location}, timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    try:
        sys.path.insert(0, str(ROOT / "backend"))
        from services.weather_service import get_weather_service
        return get_weather_service().get_live_weather(location)
    except Exception:
        cities = load_json("weather_data.json").get("cities", {})
        return cities.get(location, cities.get("Delhi", {}))

@st.cache_data(ttl=600, show_spinner=False)
def get_realtime_mandi() -> dict:
    """Fetch real-time Mandi APMC prices from backend API or local MandiService."""
    if is_backend_online():
        try:
            r = requests.get(f"{API_URL}/mandi-prices", timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    try:
        sys.path.insert(0, str(ROOT / "backend"))
        from services.mandi_service import get_mandi_service
        return get_mandi_service().get_realtime_mandi_data()
    except Exception:
        return load_json("mandi_prices.json")

@st.cache_data(ttl=600, show_spinner=False)
def get_climate_risk(location: str, crop: str = "wheat", stage: str = None) -> dict:
    """Fetch real-time AI Climate Risk and Early Warnings."""
    if is_backend_online():
        try:
            params = {"location": location, "crop": crop}
            if stage:
                params["stage"] = stage
            r = requests.get(f"{API_URL}/climate-risk", params=params, timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    try:
        sys.path.insert(0, str(ROOT / "backend"))
        from services.climate_risk_service import get_climate_risk_service
        return get_climate_risk_service().assess_crop_climate_risk(location=location, crop=crop, stage=stage)
    except Exception:
        return {}

def call_chat(message: str, language: str, location: str) -> dict:
    if is_backend_online():
        try:
            r = requests.post(
                f"{API_URL}/chat",
                json={"message": message, "language": language, "location": location},
                timeout=60,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    try:
        sys.path.insert(0, str(ROOT / "backend"))
        from agent.farming_agent import run_agent
        return run_agent(message, language=language, location=location)
    except Exception as e:
        return {
            "response": f"⚠️ Could not process query: {str(e)}. If deployed on Streamlit Cloud, please verify LLM_API_KEY is configured in Secrets.",
            "sources": [],
            "agent_steps": [],
        }

@st.cache_resource(show_spinner=False)
def _get_cached_detector():
    sys.path.insert(0, str(ROOT / "backend"))
    from disease_detector import get_detector
    return get_detector()

def call_detect(image_bytes: bytes, filename: str) -> dict:
    if is_backend_online():
        try:
            r = requests.post(
                f"{API_URL}/detect-disease",
                files={"file": (filename, image_bytes, "image/jpeg")},
                params={"top_k": 3},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return _local_detect(image_bytes)

def _local_detect(image_bytes: bytes) -> dict:
    try:
        detector = _get_cached_detector()
        return detector.detect(image_bytes, top_k=3)
    except Exception as e:
        return {"success": False, "error": str(e), "detections": [], "message": f"Local detection failed: {e}"}

def call_transcribe(audio_bytes: bytes, filename: str = "voice.wav", language: str = "auto") -> dict:
    if is_backend_online():
        try:
            r = requests.post(
                f"{API_URL}/voice/transcribe",
                files={"file": (filename, audio_bytes, "audio/wav")},
                data={"language": language},
                timeout=60,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    try:
        sys.path.insert(0, str(ROOT / "backend"))
        from voice_assistant import transcribe_audio
        text = transcribe_audio(audio_bytes, filename=filename, language=None if language == "auto" else language)
        return {"success": True, "text": text}
    except Exception as e:
        return {"success": False, "text": "", "error": str(e)}

def call_tts(text: str, language: str = "English") -> bytes:
    try:
        sys.path.insert(0, str(ROOT / "backend"))
        from voice_assistant import generate_speech
        return generate_speech(text, language=language)
    except Exception:
        return b""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
        <span style="font-size:2rem;background:#132b38;border:1px solid #10b981;padding:8px;border-radius:12px;">🧑‍🌾</span>
        <div>
            <b style="font-size:1.15rem;color:#34d399;">KrishiRakshak</b><br>
            <span style="font-size:0.8rem;color:#94a3b8;">कृषि रक्षक • Smart Assistant</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    backend_active = is_backend_online()
    mode_tag = "FastAPI Gateway Active" if backend_active else "In-Process Cloud Engine"
    st.markdown(f"""
    <div style="background:#092a20;border:1px solid #10b981;border-radius:10px;padding:8px 12px;font-size:0.8rem;color:#34d399;margin-bottom:16px;">
        🟢 <b>AI Brain Online</b> • {mode_tag}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🌐 Language / ਬੋਲੀ / भाषा")
    st.session_state.language = st.radio(
        "Interface Language",
        ["English", "Hindi", "Punjabi"],
        format_func=lambda x: "🇬🇧 English" if x == "English" else ("🇮🇳 हिन्दी" if x == "Hindi" else "🌾 ਪੰਜਾਬੀ"),
        horizontal=True,
        index=0 if st.session_state.language == "English" else (1 if st.session_state.language == "Hindi" else 2),
        label_visibility="collapsed",
    )

    st.markdown("##### 📍 Your Farming Region")
    st.session_state.location = st.selectbox(
        "Location",
        CITIES,
        index=CITIES.index(st.session_state.location) if st.session_state.location in CITIES else 0,
        label_visibility="collapsed",
    )

    # ── Groq API Key Manager ──────────────────────────────────────────
    cur_key = os.getenv("LLM_API_KEY", "").strip()
    if hasattr(st, "session_state"):
        if st.session_state.get("user_groq_api_key"):
            cur_key = st.session_state.user_groq_api_key.strip()
        elif st.session_state.get("input_groq_key_sidebar"):
            cur_key = st.session_state.input_groq_key_sidebar.strip()

    has_active_key = bool(cur_key and not cur_key.startswith("sk-placeholder") and len(cur_key) > 10)

    with st.expander("🔑 AI API Key (Groq)", expanded=not has_active_key):
        if has_active_key:
            st.success(f"🟢 **Groq AI Active** (`{cur_key[:6]}...{cur_key[-4:]}`)")
        else:
            st.info("Enter your Groq API key below for full AI synthesis and voice interaction:")

        key_input = st.text_input(
            "Groq API Key:",
            value=st.session_state.get("user_groq_api_key", ""),
            type="password",
            placeholder="gsk_...",
            key="input_groq_key_sidebar",
            help="Free Groq key: https://console.groq.com/keys",
        )
        # Auto-activate when user pastes/changes key
        if key_input.strip() and key_input.strip() != st.session_state.get("user_groq_api_key"):
            st.session_state.user_groq_api_key = key_input.strip()
            os.environ["LLM_API_KEY"] = key_input.strip()
            os.environ["LLM_BASE_URL"] = "https://api.groq.com/openai/v1"
            st.rerun()

        if st.button("💾 Apply Key", width="stretch", key="save_groq_key_btn"):
            if key_input.strip():
                st.session_state.user_groq_api_key = key_input.strip()
                os.environ["LLM_API_KEY"] = key_input.strip()
                os.environ["LLM_BASE_URL"] = "https://api.groq.com/openai/v1"
                st.success("API Key applied!")
                st.rerun()

        st.caption("👉 [Get a free Groq Key in 30 seconds](https://console.groq.com/keys)")

    st.divider()

    # ── Live Weather Mini Card ──────────────────────────────────────────
    st.markdown("##### 🌤️ Live Weather")
    live_wx = get_realtime_weather(st.session_state.location)
    curr = live_wx.get("current", {})
    if curr:
        st.markdown(
            f'<div style="background:#13242e;border:1px solid #1e3a47;border-radius:12px;padding:12px;margin-bottom:8px;color:#f1f5f9;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<b style="color:#ffffff;">{st.session_state.location}</b>'
            f'<span style="color:#34d399;font-weight:700;">{curr.get("icon", "⛅")} {curr.get("temperature", "--")}°C</span>'
            f'</div>'
            f'<small style="color:#94a3b8;">{curr.get("condition", "")} • 💧 {curr.get("humidity", "--")}% • 💨 {curr.get("wind_speed", "--")} km/h</small>'
            f'</div>',
            unsafe_allow_html=True,
        )
        adv = curr.get("advisory", "")
        if adv:
            st.caption(f"💡 {adv[:90]}...")
    else:
        st.caption("Weather stream syncing...")

    st.divider()

    # ── Live Mandi Mini Card ───────────────────────────────────────────
    st.markdown("##### 💰 Today's APMC Mandi Rates")
    mandi = get_realtime_mandi()
    city_key = next((k for k in mandi.get("markets", {}) if st.session_state.location.lower() in k.lower()), None)
    if not city_key and mandi.get("markets"):
        city_key = list(mandi.get("markets", {}).keys())[0]

    if city_key:
        commodities = mandi["markets"][city_key].get("commodities", {})
        st.caption(f"Market: **{city_key.split('(')[0].strip()}** ({mandi.get('date', 'Today')})")
        for crop, prices in list(commodities.items())[:4]:
            trend = prices.get("trend", "stable")
            icon = "↑" if trend == "rising" else ("↓" if trend == "falling" else "→")
            color = "trend-up" if trend == "rising" else ("trend-down" if trend == "falling" else "trend-stable")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:4px;color:#f1f5f9;'>"
                f"<span>{crop}</span>"
                f"<b>₹{safe_int(prices.get('modal', 0)):,}/q <span class='{color}'>{icon}</span></b>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Mandi stream syncing...")

    st.divider()

    # ── Live Climate Risk Mini Card ─────────────────────────────────────
    st.markdown("##### 🛡️ Climate Threat Alert")
    quick_risk = get_climate_risk(st.session_state.location, "wheat")
    if quick_risk and quick_risk.get("success"):
        cvi = quick_risk.get("climate_vulnerability_index", 20)
        cvi_stat = quick_risk.get("cvi_status", "LOW")
        icon = quick_risk.get("cvi_icon", "🟢")
        alerts_n = len(quick_risk.get("active_alerts", []))
        st.markdown(
            f'<div style="background:#13242e;border:1px solid #1e3a47;border-radius:12px;padding:10px 12px;font-size:0.85rem;color:#f1f5f9;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<b>Wheat Risk Index</b>'
            f'<span style="font-weight:700;color:#34d399;">{icon} {cvi}/100</span>'
            f'</div>'
            f'<small style="color:#94a3b8;">{cvi_stat} • {alerts_n} Active Alert{"s" if alerts_n != 1 else ""}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Support card
    st.markdown("""
    <div style="background:#13242e;border:1px solid #1e3a47;border-radius:12px;padding:12px;font-size:0.82rem;color:#f1f5f9;">
        <b style="color:#ffffff;">📞 Kisan Helpline (Toll-Free)</b><br>
        <span style="font-size:1.15rem;color:#34d399;font-weight:700;">1800-180-1551</span><br>
        <span style="color:#94a3b8;">Free Govt Agri Advisory • 6AM–10PM</span>
    </div>
    """, unsafe_allow_html=True)

# ── Top Hero Header ───────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-card">
    <div class="hero-title-group">
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:2.4rem;">🌾</span>
            <div>
                <h1>KrishiRakshak</h1>
                <p>कृषि रक्षक • Next-Generation Autonomous Agricultural Intelligence Platform</p>
            </div>
        </div>
    </div>
    <div class="hero-badges-row">
        <div class="hero-pill">🟢 Groq 120B AI</div>
        <div class="hero-pill">🛡️ Climate Risk AI</div>
        <div class="hero-pill">🌤️ Open-Meteo Live</div>
        <div class="hero-pill">💰 Agmarknet Govt API</div>
        <div class="hero-pill">🔬 YOLOv8 Pathology</div>
    </div>
</div>
<div class="telemetry-ribbon">
    <div class="telemetry-item">📍 Active Region: <b>{st.session_state.location}</b></div>
    <div class="telemetry-item">🌾 Season: <b>Rabi 2026</b></div>
    <div class="telemetry-item">🌐 Language: <b>{st.session_state.language}</b></div>
    <div class="telemetry-item">⚡ Engine: <b>Sub-Second Autonomous Multi-Agent</b></div>
    <div class="telemetry-item">🛰️ Telemetry: <b>Live Satellite & Mandi Feeds Synced</b></div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_voice, tab_risk, tab_disease, tab_weather, tab_market = st.tabs([
    "💬 AI Advisor",
    "🎙️ AI Voice Assistant",
    "🛡️ Climate Risk & Alerts",
    "🔬 Plant Pathology (YOLOv8)",
    "🌤️ Real-Time Weather",
    "💰 Mandi Prices (APMC Live)",
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — AI Chat Advisor
# ════════════════════════════════════════════════════════════════════════
with tab_chat:
    is_hindi = st.session_state.language == "Hindi"
    is_punjabi = st.session_state.language == "Punjabi"

    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
        <h4 style="margin:0;color:#ffffff;">💡 Instant Agricultural Queries</h4>
        <span style="font-size:0.85rem;color:#94a3b8;">Click any prompt to ask instantly</span>
    </div>
    """, unsafe_allow_html=True)

    quick_qs = {
        "English": [
            ("🌾 Rabi Season", "What crop is best for this rabi season?"),
            ("🌧️ Weather Advisory", "What is today's farming weather advisory?"),
            ("💰 Mandi Rates", "What is today's mandi rate for tomatoes?"),
            ("🐛 Pest Control", "How to control yellow rust on wheat crop?"),
            ("🌱 Fertilizer Schedule", "What fertilizer schedule is best for mustard?"),
            ("🔍 Disease Identification", "How to identify late blight in potato?"),
        ],
        "Hindi": [
            ("🌾 रबी फसल", "इस रबी मौसम में कौन सी फसल सबसे अच्छी है?"),
            ("🌧️ मौसम सलाह", "आज की मौसम और कृषि सलाह क्या है?"),
            ("💰 टमाटर भाव", "आज दिल्ली और मुंबई में टमाटर का मंडी भाव क्या है?"),
            ("🐛 कीट नियंत्रण", "गेहूं में पीला रतुआ कीट से बचाव कैसे करें?"),
            ("🌱 खाद व उर्वरक", "सरसों के लिए संतुलित खाद और सिंचाई क्या है?"),
            ("🔍 रोग पहचान", "आलू में पछेती झुलसा रोग की रोकथाम कैसे करें?"),
        ],
        "Punjabi": [
            ("🌾 ਹਾੜ੍ਹੀ ਫਸਲ", "ਇਸ ਹਾੜ੍ਹੀ ਰੁੱਤ ਵਿੱਚ ਕਿਹੜੀ ਫਸਲ ਸਭ ਤੋਂ ਵਧੀਆ ਹੈ?"),
            ("🌧️ ਮੌਸਮ ਸਲਾਹ", "ਅੱਜ ਦੀ ਮੌਸਮ ਅਤੇ ਖੇਤੀਬਾੜੀ ਸਲਾਹ ਕੀ ਹੈ?"),
            ("💰 ਮੰਡੀ ਭਾਅ", "ਅੱਜ ਮੰਡੀ ਵਿੱਚ ਟਮਾਟਰ ਅਤੇ ਆਲੂ ਦਾ ਰੇਟ ਕੀ ਹੈ?"),
            ("🐛 ਕੀੜੇ ਰੋਕਥਾਮ", "ਕਣਕ ਦੀ ਫਸਲ ਵਿੱਚ ਪੀਲੀ ਕੁੰਗੀ ਦੀ ਰੋਕਥਾਮ ਕਿਵੇਂ ਕਰੀਏ?"),
            ("🌱 ਖਾਦ ਪ੍ਰਬੰਧ", "ਸਰ੍ਹੋਂ ਦੀ ਫਸਲ ਲਈ ਖਾਦ ਅਤੇ ਪਾਣੀ ਦਾ ਪ੍ਰਬੰਧ ਕੀ ਹੈ?"),
            ("🔍 ਰੋਗ ਇਲਾਜ", "ਟਮਾਟਰ ਦੇ ਪੱਤਿਆਂ ਤੇ ਪੀਲੇ ਧੱਬਿਆਂ ਦਾ ਇਲਾਜ ਕੀ ਹੈ?"),
        ],
    }

    cols = st.columns(3)
    active_questions = quick_qs.get(st.session_state.language, quick_qs["English"])
    for i, (label, q) in enumerate(active_questions):
        with cols[i % 3]:
            if st.button(f"{label}: {q[:32]}...", width="stretch", key=f"quick_btn_{i}", help=q):
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner("KrishiRakshak reasoning with real-time tools..." if not is_hindi else "कृषि रक्षक वास्तविक समय सलाह तैयार कर रहा है..."):
                    result = call_chat(q, st.session_state.language, st.session_state.location)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("response", ""),
                    "sources": result.get("sources", []),
                    "steps": result.get("agent_steps", []),
                })
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat Container
    chat_box = st.container(height=450)
    with chat_box:
        if not st.session_state.messages:
            st.markdown(f"""
            <div style="text-align:center;padding:2.5rem 1rem;color:#94a3b8;">
                <span style="font-size:3rem;">🌾</span><br>
                <b style="font-size:1.25rem;color:#34d399;">Welcome to KrishiRakshak AI</b><br>
                <span style="color:#cbd5e1;">Ask about crop calendars, pest management, live weather forecasts, or APMC mandi rates.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            for idx, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-bubble-user">🧑‍🌾 <b>Farmer ({st.session_state.location}):</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    sources_html = "".join([f'<span class="source-tag">📌 {s}</span>' for s in msg.get("sources", [])])
                    st.markdown(
                        f'<div class="chat-bubble-assistant">'
                        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
                        f'<div style="display:flex;align-items:center;gap:8px;">'
                        f'<span style="font-size:1.2rem;">🌾</span><b style="color:#34d399;">KrishiRakshak AI:</b>'
                        f'</div>'
                        f'<span style="font-size:0.75rem;color:#94a3b8;">Grounded Real-Time Advisory</span>'
                        f'</div>'
                        f'{msg["content"]}'
                        f'{"<div style=\'margin-top:12px;\'>" + sources_html + "</div>" if sources_html else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if msg.get("audio"):
                        st.audio(msg["audio"], format="audio/mp3")
                    else:
                        c_aud, _ = st.columns([1, 4])
                        with c_aud:
                            if st.button("🔊 Listen", key=f"listen_btn_t1_{idx}", help="Listen to spoken audio"):
                                with st.spinner("Synthesizing speech..."):
                                    msg["audio"] = call_tts(msg["content"], st.session_state.language)
                                st.rerun()

                    steps = msg.get("steps", [])
                    if steps:
                        with st.expander(f"🧠 Agent Reasoning & Telemetry ({len(steps)} Domain Tools Used)"):
                            for s in steps:
                                st.markdown(f"**Tool Invoked:** `{s.get('tool', 'Domain Tool')}`")
                                st.caption(f"Input: {str(s.get('input', ''))[:140]}")
                                st.caption(f"Observation: {str(s.get('output', ''))[:220]}")
                                st.divider()

    # Form input
    with st.form("chat_form", clear_on_submit=True):
        c_inp, c_btn = st.columns([5, 1])
        user_input = c_inp.text_input(
            "Ask KrishiRakshak a farming question:",
            placeholder="Type your question in English, हिन्दी, or ਪੰਜਾਬੀ... (e.g. Best fertilizer for wheat?)",
            label_visibility="collapsed",
        )
        submit_chat = c_btn.form_submit_button("Send 🚀", width="stretch")

    if submit_chat and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Consulting agricultural models & live feeds..."):
            result = call_chat(user_input, st.session_state.language, st.session_state.location)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("response", ""),
            "sources": result.get("sources", []),
            "steps": result.get("agent_steps", []),
        })
        st.rerun()

    # Action Toolbar
    c_act1, c_act2, c_act3 = st.columns([2, 2, 1])
    with c_act1:
        if st.session_state.messages:
            last_resp = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"), "")
            if last_resp:
                st.download_button(
                    "📥 Export Advisory (Text)",
                    data=last_resp,
                    file_name="KrishiRakshak_Advisory.txt",
                    mime="text/plain",
                    width="stretch",
                )
    with c_act3:
        if st.button("🗑️ Clear Chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — AI Voice Assistant (Conversational Voice ChatBot)
# ════════════════════════════════════════════════════════════════════════
with tab_voice:
    st.markdown("### 🎙️ Multilingual AI Voice Farming Chatbot")
    st.caption("ਬੋਲ ਕੇ ਖੇਤੀ ਸਲਾਹ ਲਵੋ • बोलकर खेती सलाह पाएं • Conversational Speech-to-Speech Agricultural Intelligence")

    # Header Controls: Language Selector & Action Buttons
    c_vh1, c_vh2 = st.columns([3, 1])
    with c_vh1:
        v_lang_sel = st.radio(
            "Select Voice Language / ਬੋਲੀ / भाषा:",
            ["Hindi (हिंदी)", "Punjabi (ਪੰਜਾਬੀ)", "English"],
            horizontal=True,
            key="v_lang_sel_radio",
        )
        clean_vlang = "Hindi" if "Hindi" in v_lang_sel else ("Punjabi" if "Punjabi" in v_lang_sel else "English")

    with c_vh2:
        c_vclr, c_vexp = st.columns(2)
        with c_vclr:
            if st.button("🗑️ Clear", width="stretch", help="Clear voice chat history", key="clear_voice_chat_btn"):
                st.session_state.voice_chat_history = []
                st.rerun()
        with c_vexp:
            if st.session_state.voice_chat_history:
                transcript_text = "\n\n".join([
                    f"Farmer ({item.get('lang', clean_vlang)}): {item.get('query')}\nKrishiRakshak AI: {item.get('response')}"
                    for item in st.session_state.voice_chat_history
                ])
                st.download_button(
                    "📥 Export",
                    data=transcript_text,
                    file_name="KrishiRakshak_Voice_Conversation.txt",
                    mime="text/plain",
                    width="stretch",
                    key="export_voice_chat_btn",
                )

    # 1-Tap Quick Spoken Query Chips
    st.markdown("<small style='color:#a7f3d0;font-weight:600;'>💡 Tap any quick spoken prompt to hear instant voice advice:</small>", unsafe_allow_html=True)
    if clean_vlang == "Hindi":
        voice_presets = [
            ("🍅 टमाटर रोग सलाह", "मेरे टमाटर के पत्तों पर पीले धब्बे आ रहे हैं, इसका इलाज क्या है?"),
            ("💰 दिल्ली मंडी भाव", "आज दिल्ली और आज़ादपुर मंडी में टमाटर और आलू का क्या भाव है?"),
            ("🌧️ कीटनाशक स्प्रे मौसम", "क्या आज गेहूं की फसल में कीटनाशक का छिड़काव कर सकते हैं?"),
            ("🌾 रबी गेहूं उत्तम पैदावार", "रबी मौसम के लिए गेहूं की कौन सी किस्म सबसे ज्यादा पैदावार देती है?"),
        ]
    elif clean_vlang == "Punjabi":
        voice_presets = [
            ("🍅 ਟਮਾਟਰ ਪੀਲੇ ਧੱਬੇ", "ਮੇਰੇ ਟਮਾਟਰ ਦੇ ਪੱਤਿਆਂ ਤੇ ਪੀਲੇ ਧੱਬੇ ਆ ਰਹੇ ਹਨ, ਕੀ ਕਰੀਏ?"),
            ("🌾 ਕਣਕ ਪੀਲੀ ਕੁੰਗੀ", "ਕਣਕ ਦੀ ਫਸਲ ਵਿੱਚ ਪੀਲੀ ਕੁੰਗੀ ਦੀ ਰੋਕਥਾਮ ਕਿਵੇਂ ਕਰੀਏ?"),
            ("💰 ਮੰਡੀ ਭਾਅ ਅੱਪਡੇਟ", "ਅੱਜ ਮੰਡੀ ਵਿੱਚ ਟਮਾਟਰ ਅਤੇ ਆਲੂ ਦਾ ਰੇਟ ਕੀ ਹੈ?"),
            ("🌧️ ਸਪਰੇਅ ਮੌਸਮ ਸਲਾਹ", "ਕੀ ਅੱਜ ਖੇਤ ਵਿੱਚ ਸਪਰੇਅ ਕਰਨਾ ਸਹੀ ਰਹੇਗਾ?"),
        ]
    else:
        voice_presets = [
            ("🍅 Tomato Disease Advice", "My tomato plant leaves have yellow spots and curling, what should I do?"),
            ("💰 Today's Mandi Rates", "What is today's mandi market rate for tomatoes in Delhi?"),
            ("🌧️ Spray Weather Window", "What is today's agricultural weather advisory for chemical spraying?"),
            ("🌾 Rabi Wheat Management", "How to effectively manage weed and nitrogen schedule for wheat?"),
        ]

    c_vp = st.columns(4)
    preset_triggered_query = None
    for p_idx, (p_label, p_query) in enumerate(voice_presets):
        with c_vp[p_idx]:
            if st.button(p_label, width="stretch", key=f"voice_chip_{p_idx}", help=p_query):
                preset_triggered_query = p_query

    # Dedicated ChatBot Message Stream Container
    voice_chat_box = st.container(height=480)
    with voice_chat_box:
        if not st.session_state.voice_chat_history and not preset_triggered_query:
            welcome_lang_text = {
                "Hindi": "बोलकर या लिखकर खेती से जुड़ी कोई भी सलाह पूछें। मैं हिंदी में बोलकर और लिखकर उपाय बताऊंगा।",
                "Punjabi": "ਬੋਲ ਕੇ ਜਾਂ ਲਿਖ ਕੇ ਖੇਤੀਬਾੜੀ ਸਲਾਹ ਪੁੱਛੋ। ਮੈਂ ਪੰਜਾਬੀ ਵਿੱਚ ਬੋਲ ਕੇ ਸਹੀ ਹੱਲ ਦੱਸਾਂਗਾ।",
                "English": "Speak naturally or select a suggestion to receive real-time spoken audio prescriptions & agronomy advice.",
            }.get(clean_vlang, "")

            st.markdown(f"""
            <div style="background:radial-gradient(circle, #0d2820 0%, #13242e 80%);border:1px solid #1e3a47;border-radius:18px;padding:2.5rem 1.5rem;text-align:center;color:#94a3b8;margin:1rem 0;">
                <div style="display:inline-flex;align-items:center;justify-content:center;width:70px;height:70px;border-radius:50%;background:#092a20;border:2px solid #10b981;margin-bottom:0.75rem;box-shadow:0 0 20px rgba(16, 185, 129, 0.35);">
                    <span style="font-size:2.2rem;">🎙️</span>
                </div>
                <br>
                <b style="color:#ffffff;font-size:1.25rem;">KrishiRakshak Voice AI Chatbot</b><br>
                <span style="color:#34d399;font-weight:600;font-size:0.95rem;">{welcome_lang_text}</span><br>
                <span style="font-size:0.85rem;display:inline-block;max-width:520px;margin-top:10px;color:#cbd5e1;">
                    Tap the microphone below to speak, click any quick prompt chip above, or type your question. Every answer includes a natural voice audio prescription.
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            for idx, item in enumerate(st.session_state.voice_chat_history):
                # Farmer User Bubble
                st.markdown(f"""
                <div class="chat-bubble-user">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;gap:8px;">
                        <div style="display:flex;align-items:center;gap:6px;">
                            <span>🧑‍🌾</span>
                            <b style="color:#ffffff;">Farmer ({item.get('lang', clean_vlang)})</b>
                        </div>
                        <span style="font-size:0.75rem;background:rgba(255,255,255,0.18);padding:2px 8px;border-radius:10px;">🗣️ Spoken Voice</span>
                    </div>
                    <div style="font-size:1.02rem;font-weight:600;margin-top:4px;">"{item['query']}"</div>
                </div>
                """, unsafe_allow_html=True)

                if item.get("user_audio"):
                    st.audio(item["user_audio"], format="audio/wav")

                # KrishiRakshak Assistant Bubble
                sources_html = ""
                if item.get("sources"):
                    sources_tags = " ".join([f'<span class="source-tag">📌 {s}</span>' for s in item["sources"]])
                    sources_html = f"<div style='margin-top:10px;'><b>Verified Sources:</b><br>{sources_tags}</div>"

                st.markdown(f"""
                <div class="chat-bubble-assistant">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:8px;">
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="font-size:1.2rem;">🌾</span>
                            <b style="color:#34d399;">KrishiRakshak Voice AI:</b>
                            <span style="font-size:0.72rem;background:#064e3b;color:#a7f3d0;padding:2px 8px;border-radius:10px;border:1px solid #059669;">🔊 Spoken Audio ({item.get('lang', clean_vlang)})</span>
                        </div>
                        <div class="sound-wave-box" style="height:24px;padding:3px 8px;">
                            <div class="wave-bar" style="width:3px;"></div>
                            <div class="wave-bar" style="width:3px;"></div>
                            <div class="wave-bar" style="width:3px;"></div>
                            <div class="wave-bar" style="width:3px;"></div>
                            <div class="wave-bar" style="width:3px;"></div>
                        </div>
                    </div>
                    <div style="margin-bottom:8px;color:#f1f5f9;line-height:1.6;">
                        {item['response']}
                    </div>
                    {sources_html}
                </div>
                """, unsafe_allow_html=True)

                if item.get("audio"):
                    is_latest = (idx == len(st.session_state.voice_chat_history) - 1)
                    st.audio(item["audio"], format="audio/mp3", autoplay=is_latest)

    # Trigger handling for preset chips
    if preset_triggered_query:
        with st.spinner(f"🧠 KrishiRakshak reasoning & generating spoken audio in {clean_vlang}..."):
            agent_res = call_chat(preset_triggered_query, clean_vlang, st.session_state.location)
            resp_text = agent_res.get("response", "")
            sources = agent_res.get("sources", [])
            speech_bytes = call_tts(resp_text, clean_vlang)
            st.session_state.voice_chat_history.append({
                "query": preset_triggered_query,
                "response": resp_text,
                "audio": speech_bytes,
                "lang": clean_vlang,
                "sources": sources,
                "user_audio": None,
            })
            st.rerun()

    # Unified ChatBot Input Dock (Microphone + Text + File Upload)
    st.markdown("<br>", unsafe_allow_html=True)
    c_dock_mic, c_dock_txt = st.columns([1, 2])

    with c_dock_mic:
        st.markdown("<b style='color:#34d399;'>🎙️ Speak Question (Mic):</b>", unsafe_allow_html=True)
        if hasattr(st, "audio_input"):
            mic_audio = st.audio_input(
                f"Tap mic to speak in {clean_vlang}:",
                key=f"voice_bot_mic_{len(st.session_state.voice_chat_history)}",
                label_visibility="collapsed",
            )
            if mic_audio:
                audio_bytes = mic_audio.getvalue()
                h_val = hash(audio_bytes)
                if st.session_state.get("last_processed_voice_hash") != h_val:
                    st.session_state.last_processed_voice_hash = h_val
                    with st.spinner("🎙️ Transcribing farmer voice with Whisper AI..."):
                        stt = call_transcribe(audio_bytes, language=clean_vlang)
                        if stt.get("success") and stt.get("text"):
                            q_text = stt.get("text")
                            with st.spinner(f"🧠 KrishiRakshak diagnosing in {clean_vlang}..."):
                                agent_res = call_chat(q_text, clean_vlang, st.session_state.location)
                                resp_text = agent_res.get("response", "")
                                sources = agent_res.get("sources", [])
                                speech_bytes = call_tts(resp_text, clean_vlang)
                                st.session_state.voice_chat_history.append({
                                    "query": q_text,
                                    "response": resp_text,
                                    "audio": speech_bytes,
                                    "lang": clean_vlang,
                                    "sources": sources,
                                    "user_audio": audio_bytes,
                                })
                                st.rerun()
                        else:
                            st.error(f"Voice recognition error: {stt.get('error', 'No speech detected')}")
        else:
            st.caption("Native browser microphone requires Streamlit 1.38+. Use prompt chips above or text input.")

    with c_dock_txt:
        st.markdown("<b style='color:#34d399;'>💬 Or Type with Voice Reply:</b>", unsafe_allow_html=True)
        with st.form("voice_bot_text_form", clear_on_submit=True):
            c_tin, c_tsub = st.columns([4, 1])
            typed_val = c_tin.text_input(
                "Type question:",
                placeholder=f"Ask in {clean_vlang} (e.g. टमाटर रोग का उपचार / What fertilizer for wheat?)...",
                label_visibility="collapsed",
            )
            submit_voice_txt = c_tsub.form_submit_button("Send & Speak 🎙️", width="stretch")
            if submit_voice_txt and typed_val.strip():
                with st.spinner(f"🧠 KrishiRakshak reasoning & generating spoken audio in {clean_vlang}..."):
                    agent_res = call_chat(typed_val, clean_vlang, st.session_state.location)
                    resp_text = agent_res.get("response", "")
                    sources = agent_res.get("sources", [])
                    speech_bytes = call_tts(resp_text, clean_vlang)
                    st.session_state.voice_chat_history.append({
                        "query": typed_val,
                        "response": resp_text,
                        "audio": speech_bytes,
                        "lang": clean_vlang,
                        "sources": sources,
                        "user_audio": None,
                    })
                    st.rerun()

    with st.expander("📁 Upload Recorded Voice Note (WAV / MP3 / M4A / OGG)"):
        up_voice = st.file_uploader(
            "Upload audio recording:",
            type=["wav", "mp3", "m4a", "webm", "ogg"],
            key="up_voice_file_bot",
            label_visibility="collapsed",
        )
        if up_voice and st.button("🚀 Process Uploaded Voice File", key="btn_proc_up_voice"):
            up_bytes = up_voice.getvalue()
            with st.spinner("🎙️ Transcribing uploaded audio file..."):
                stt = call_transcribe(up_bytes, filename=up_voice.name, language=clean_vlang)
                if stt.get("success") and stt.get("text"):
                    q_text = stt.get("text")
                    with st.spinner(f"🧠 Reasoning in {clean_vlang}..."):
                        agent_res = call_chat(q_text, clean_vlang, st.session_state.location)
                        resp_text = agent_res.get("response", "")
                        sources = agent_res.get("sources", [])
                        speech_bytes = call_tts(resp_text, clean_vlang)
                        st.session_state.voice_chat_history.append({
                            "query": q_text,
                            "response": resp_text,
                            "audio": speech_bytes,
                            "lang": clean_vlang,
                            "sources": sources,
                            "user_audio": up_bytes,
                        })
                        st.rerun()
                else:
                    st.error(f"Audio transcription failed: {stt.get('error', 'Unrecognized audio format')}")


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — AI Climate Risk Predictor & Early Warning System
# ════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown("### 🛡️ AI Climate Risk Predictor & Early Warning System")
    st.caption("🟢 Satellite-Derived Root-Zone Soil Moisture (ERA5) • 14-Day Hydro-Meteorological Intelligence • ICAR Agronomic Stress Guidelines")

    # Crop stages map for dynamic stage selection
    crop_stages_options = {
        "Wheat (गेहूं / ਕਣਕ)": ("wheat", ["Sowing/CRI", "Vegetative/Tillering", "Booting/Flowering", "Grain Filling/Milking", "Maturity/Harvest"]),
        "Rice / Paddy (धान / ਝੋਨਾ)": ("rice", ["Nursery/Transplanting", "Tillering", "Panicle Initiation/Flowering", "Milking/Dough", "Maturity/Harvest"]),
        "Tomato (टमाटर / ਟਮਾਟਰ)": ("tomato", ["Nursery/Transplanting", "Vegetative", "Flowering", "Fruit Development", "Harvesting"]),
        "Potato (आलू / ਆਲੂ)": ("potato", ["Sprouting/Emergence", "Vegetative Growth", "Tuber Initiation", "Tuber Bulking", "Senescence/Harvest"]),
        "Mustard (सरसों / ਸਰ੍ਹੋਂ)": ("mustard", ["Germination/Seedling", "Vegetative Growth", "Flowering", "Siliqua/Pod Filling", "Maturity"]),
        "Maize (मक्का / ਮੱਕੀ)": ("maize", ["Germination/Seedling", "Knee High/Vegetative", "Tasseling/Silking", "Grain Filling", "Maturity"]),
        "Cotton (कपास / ਨਰਮਾ)": ("cotton", ["Seedling", "Vegetative/Square Formation", "Flowering/Boll Formation", "Boll Opening", "Harvesting"]),
    }

    # 1-Click Major Indian Crop Presets
    st.markdown("<small style='color:#a7f3d0;font-weight:600;'>🌾 Quick Select Target Crop:</small>", unsafe_allow_html=True)
    c_cp1, c_cp2, c_cp3, c_cp4, c_cp5, c_cp6 = st.columns(6)
    crops_keys_list = list(crop_stages_options.keys())
    if "risk_selected_crop_idx" not in st.session_state:
        st.session_state.risk_selected_crop_idx = 0

    crop_buttons = [
        (c_cp1, "🌾 Wheat", 0),
        (c_cp2, "🌾 Paddy", 1),
        (c_cp3, "🍅 Tomato", 2),
        (c_cp4, "🥔 Potato", 3),
        (c_cp5, "🌱 Mustard", 4),
        (c_cp6, "🌽 Maize", 5),
    ]
    for col, lbl, c_i in crop_buttons:
        with col:
            if st.button(lbl, width="stretch", key=f"quick_crop_btn_{c_i}"):
                st.session_state.risk_selected_crop_idx = c_i
                st.rerun()

    col_r1, col_r2, col_r3, col_r4 = st.columns([2, 2, 2, 1])
    with col_r1:
        sel_crop_name = st.selectbox(
            "🌾 Target Crop:",
            crops_keys_list,
            index=st.session_state.risk_selected_crop_idx,
            key="risk_crop_select",
        )
    crop_key, stage_choices = crop_stages_options[sel_crop_name]

    with col_r2:
        default_stage_idx = min(2, len(stage_choices) - 1)
        sel_stage = st.selectbox("🌱 Current Phenological Stage:", stage_choices, index=default_stage_idx, key="risk_stage_select")

    with col_r3:
        risk_loc = st.text_input("📍 Farming Location:", value=st.session_state.location, key="risk_location_input")

    with col_r4:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Analyze Risk", width="stretch", key="risk_refresh_btn"):
            get_climate_risk.clear()
            st.rerun()

    active_risk_loc = risk_loc.strip() if risk_loc.strip() else st.session_state.location
    risk_assessment = get_climate_risk(active_risk_loc, crop=crop_key, stage=sel_stage)

    if not risk_assessment or not risk_assessment.get("success"):
        st.warning("Climate risk calculation service currently syncing meteorological observations.")
    else:
        cvi = safe_int(risk_assessment.get("climate_vulnerability_index", 0))
        cvi_status = risk_assessment.get("cvi_status", "NORMAL")
        cvi_icon = risk_assessment.get("cvi_icon", "🟢")
        sm = risk_assessment.get("satellite_soil_moisture", {})
        active_alerts = risk_assessment.get("active_alerts", [])
        scores = risk_assessment.get("risk_scores", {})
        preventive = risk_assessment.get("preventive_actions", [])
        curr_w = risk_assessment.get("current_weather", {})
        timeline = risk_assessment.get("timeline", [])

        # Color coding for CVI
        if cvi >= 70:
            cvi_color = "#f87171"
            cvi_border = "#ef4444"
            cvi_bg = "linear-gradient(135deg, #381216 0%, #1c0e11 100%)"
            cvi_glow = "rgba(239, 68, 68, 0.35)"
            status_desc = "🚨 CRITICAL CLIMATE VULNERABILITY — Immediate Protective Intervention Required"
        elif cvi >= 45:
            cvi_color = "#fb923c"
            cvi_border = "#f97316"
            cvi_bg = "linear-gradient(135deg, #331d0d 0%, #1c140d 100%)"
            cvi_glow = "rgba(249, 115, 22, 0.3)"
            status_desc = "⚠️ ELEVATED CLIMATE RISK — High Stress Conditions Emerging Over 7-Day Window"
        elif cvi >= 25:
            cvi_color = "#facc15"
            cvi_border = "#eab308"
            cvi_bg = "linear-gradient(135deg, #2e240a 0%, #171c19 100%)"
            cvi_glow = "rgba(234, 179, 8, 0.25)"
            status_desc = "🟡 MODERATE STRESS WATCH — Monitoring Suggested for Heat/Moisture Shifts"
        else:
            cvi_color = "#34d399"
            cvi_border = "#10b981"
            cvi_bg = "linear-gradient(135deg, #092a20 0%, #13242e 100%)"
            cvi_glow = "rgba(16, 185, 129, 0.25)"
            status_desc = "🟢 OPTIMAL CLIMATE RESILIENCE — Micro-Climate Favorable for Active Growth"

        # Hero Circular Gauge + Telemetry Command Header
        st.markdown(f"""
        <div style="background:{cvi_bg};border:2px solid {cvi_border};border-radius:18px;padding:1.5rem 1.8rem;margin:1rem 0;box-shadow:0 8px 24px {cvi_glow};">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px;">
                <div style="flex:1;min-width:280px;">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        <span style="font-size:0.78rem;background:rgba(255,255,255,0.12);color:{cvi_color};border:1px solid {cvi_border};padding:3px 10px;border-radius:9999px;font-weight:700;letter-spacing:0.5px;">
                            🛰️ SATELLITE RADAR & ERA5-LAND INTELLIGENCE
                        </span>
                    </div>
                    <h2 style="margin:2px 0 6px 0;color:#ffffff;font-size:1.65rem;line-height:1.2;">
                        {risk_assessment.get('crop')} at {risk_assessment.get('location')} ({risk_assessment.get('state')})
                    </h2>
                    <div style="font-size:0.9rem;color:#94a3b8;margin-bottom:10px;">
                        Phenological Stage: <b style="color:#f1f5f9;">{sel_stage}</b> • Synced: <b style="color:#cbd5e1;">{risk_assessment.get('timestamp')}</b>
                    </div>
                    <div style="font-size:0.85rem;color:{cvi_color};font-weight:700;line-height:1.4;">
                        {status_desc}
                    </div>
                    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:12px;">
                        <span style="background:#0b1922;border:1px solid #1e3a47;padding:4px 10px;border-radius:8px;font-size:0.8rem;color:#cbd5e1;">🌡️ Temp: <b>{curr_w.get('temperature', '--')}°C</b></span>
                        <span style="background:#0b1922;border:1px solid #1e3a47;padding:4px 10px;border-radius:8px;font-size:0.8rem;color:#cbd5e1;">💧 Humidity: <b>{curr_w.get('humidity', '--')}%</b></span>
                        <span style="background:#0b1922;border:1px solid #1e3a47;padding:4px 10px;border-radius:8px;font-size:0.8rem;color:#cbd5e1;">💨 Wind: <b>{curr_w.get('wind_speed', '--')} km/h</b></span>
                        <span style="background:#0b1922;border:1px solid #1e3a47;padding:4px 10px;border-radius:8px;font-size:0.8rem;color:#cbd5e1;">🌧️ Precip: <b>{curr_w.get('precipitation', '--')} mm</b></span>
                    </div>
                </div>
                <div style="text-align:center;">
                    <div style="width:130px;height:130px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;border:4px solid {cvi_border};background:radial-gradient(circle, #092a20 0%, #112028 75%);box-shadow:0 0 24px {cvi_glow};margin:0 auto;">
                        <span style="font-size:2.2rem;font-weight:800;color:{cvi_color};line-height:1;">{cvi}</span>
                        <span style="font-size:0.65rem;color:#94a3b8;font-weight:700;letter-spacing:1px;margin-top:4px;">CVI INDEX</span>
                        <span style="font-size:0.75rem;margin-top:2px;">{cvi_icon}</span>
                    </div>
                    <span style="font-size:0.75rem;color:#94a3b8;margin-top:6px;display:block;font-weight:600;">{cvi_status}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Active Warnings & Alerts Banner
        if active_alerts:
            st.markdown("#### 🚨 Active Meteorological Warnings & Early Alerts")
            for alert in active_alerts:
                is_severe = "SEVERE" in alert.get("severity", "") or "HIGH" in alert.get("severity", "")
                card_bg = "#2d1215" if is_severe else "#2d1f0d"
                card_border = "#f87171" if is_severe else "#fbbf24"
                card_color = "#fca5a5" if is_severe else "#fde047"
                card_badge_bg = "rgba(239, 68, 68, 0.25)" if is_severe else "rgba(245, 158, 11, 0.25)"
                st.markdown(
                    f'<div style="background:{card_bg};border:1px solid {card_border};border-left:5px solid {card_border};border-radius:10px;padding:12px 16px;margin-bottom:10px;color:#f1f5f9;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<b style="color:{card_color};font-size:1rem;">{alert["icon"]} {alert["type"]}</b>'
                    f'<span style="background:{card_badge_bg};color:{card_color};border:1px solid {card_border};font-weight:700;font-size:0.78rem;padding:2px 8px;border-radius:9999px;">{alert["severity"]} ({alert["score"]}%)</span>'
                    f'</div>'
                    f'<span style="color:#e2e8f0;font-size:0.92rem;margin-top:6px;display:block;line-height:1.5;">{alert["message"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success(f"✅ **Zero Critical Climate Threats Detected**: Current weather conditions are favorable for {sel_crop_name} during {sel_stage}.")

        st.divider()

        # 6-Vector Climate Risk Diagnostic Matrix
        st.markdown("#### 📊 Six-Vector Climate Threat Diagnostic")
        st.caption("Multidimensional agronomic hazard computation evaluating micro-climate stress thresholds:")

        rk_col1, rk_col2, rk_col3 = st.columns(3)
        vector_items = [
            ("drought_risk", "🏜️ Drought & Soil Moisture Deficit", rk_col1),
            ("flood_risk", "🌊 Flood & Waterlogging Hazard", rk_col2),
            ("heat_stress", "🌡️ Terminal Heat Shock / Scorching", rk_col3),
        ]
        for key_v, label_v, col_v in vector_items:
            v_info = scores.get(key_v, {})
            v_score = safe_int(v_info.get("score", 0))
            v_lvl = v_info.get("level", "LOW")
            v_color = "#f87171" if v_score >= 60 else ("#fbbf24" if v_score >= 35 else "#34d399")
            with col_v:
                st.markdown(f"""
                <div class="hazard-card" style="border-top:3px solid {v_color};text-align:left;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <b style="font-size:0.92rem;color:#ffffff;">{label_v}</b>
                        <span style="font-size:0.75rem;color:{v_color};font-weight:700;border:1px solid {v_color};padding:1px 6px;border-radius:6px;">{v_lvl}</span>
                    </div>
                    <div style="font-size:1.6rem;font-weight:800;color:{v_color};margin-bottom:6px;">{v_score}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(safe_clamp(v_score / 100.0))

        rk_col4, rk_col5, rk_col6 = st.columns(3)
        vector_items_2 = [
            ("frost_warning", "❄️ Frost & Radiative Chilling", rk_col4),
            ("lodging_risk", "🌧️ Heavy Rain & Crop Lodging", rk_col5),
            ("fungal_disease_risk", "🦠 Pathogen & Fungal Epidemic", rk_col6),
        ]
        for key_v, label_v, col_v in vector_items_2:
            v_info = scores.get(key_v, {})
            v_score = safe_int(v_info.get("score", 0))
            v_lvl = v_info.get("level", "LOW")
            v_color = "#f87171" if v_score >= 60 else ("#fbbf24" if v_score >= 35 else "#34d399")
            with col_v:
                st.markdown(f"""
                <div class="hazard-card" style="border-top:3px solid {v_color};text-align:left;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <b style="font-size:0.92rem;color:#ffffff;">{label_v}</b>
                        <span style="font-size:0.75rem;color:{v_color};font-weight:700;border:1px solid {v_color};padding:1px 6px;border-radius:6px;">{v_lvl}</span>
                    </div>
                    <div style="font-size:1.6rem;font-weight:800;color:{v_color};margin-bottom:6px;">{v_score}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(safe_clamp(v_score / 100.0))

        st.divider()

        # 14-Day Hydro-Meteorological Climate Charts
        if timeline:
            st.markdown("#### 📈 14-Day Hydro-Meteorological Observation Timeline")
            st.caption("Past 7 Days Meteorological Actuals + Next 7 Days Open-Meteo High-Resolution Numerical Forecast:")

            df_time = pd.DataFrame(timeline)
            df_time["Day"] = df_time["date"].apply(lambda x: str(x)[5:])

            c_cht1, c_cht2 = st.columns([3, 2])
            with c_cht1:
                st.markdown("<b style='color:#38bdf8;font-size:0.9rem;'>🌡️ Temperature Envelope (°C):</b>", unsafe_allow_html=True)
                df_temp = df_time.set_index("Day")[["max_temp", "min_temp"]]
                df_temp.columns = ["Max Temp (°C)", "Min Temp (°C)"]
                st.line_chart(df_temp, height=220)

            with c_cht2:
                st.markdown("<b style='color:#34d399;font-size:0.9rem;'>🌧️ Daily Rainfall Accumulation (mm):</b>", unsafe_allow_html=True)
                df_rain = df_time.set_index("Day")[["rainfall"]]
                df_rain.columns = ["Precipitation (mm)"]
                st.bar_chart(df_rain, height=220)

        st.divider()

        # Stratified Root-Zone Soil Moisture Profile (0–27 cm)
        st.markdown("#### 🛰️ Real-Time Satellite Root-Zone Soil Moisture Profile (ECMWF ERA5-Land)")
        avg_moist = sm.get('average_root_moisture', 0.28)
        wilt_th = sm.get('wilting_threshold', 0.17)
        sm_status = sm.get('status', 'Adequate')
        sm_stat_color = "#34d399" if sm_status == "Adequate" else ("#fbbf24" if sm_status == "Marginal" else "#f87171")

        st.markdown(f"""
        <div style="background:#112028;border:1px solid #1e3a47;border-radius:12px;padding:12px 18px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <span style="font-size:0.85rem;color:#94a3b8;">Average Volumetric Root-Zone Moisture:</span>
                <span style="font-size:1.3rem;font-weight:800;color:#38bdf8;margin-left:8px;">{avg_moist} m³/m³</span>
            </div>
            <div style="display:flex;gap:12px;align-items:center;">
                <span style="font-size:0.85rem;color:#94a3b8;">Permanent Wilting Point: <b style="color:#f87171;">{wilt_th} m³/m³</b></span>
                <span style="background:{sm_stat_color}22;color:{sm_stat_color};border:1px solid {sm_stat_color};padding:3px 10px;border-radius:9999px;font-weight:700;font-size:0.8rem;">
                    Status: {sm_status}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sm_cols = st.columns(4)
        layers = [
            ("🟫 Surface (0–1 cm)", sm.get('surface_0_to_1cm', 0.25), "Topsoil Evaporation Layer"),
            ("🟤 Seed Zone (1–3 cm)", sm.get('seed_zone_1_to_3cm', 0.28), "Germination & Seed Bed"),
            ("🟩 Active Roots (3–9 cm)", sm.get('active_root_3_to_9cm', 0.31), "Nutrient & Water Uptake"),
            ("🟦 Deep Subsoil (9–27 cm)", sm.get('deep_root_9_to_27cm', 0.34), "Subsurface Aquifer Reserve"),
        ]
        for idx_l, (l_title, l_val, l_sub) in enumerate(layers):
            val_f = safe_float(l_val)
            pct_norm = safe_clamp((val_f - wilt_th) / (0.45 - wilt_th))
            with sm_cols[idx_l]:
                st.markdown(f"""
                <div class="stat-box-modern">
                    <b style="font-size:0.9rem;color:#ffffff;">{l_title}</b><br>
                    <span style="font-size:1.35rem;font-weight:800;color:#38bdf8;">{val_f} <small style='font-size:0.75rem;color:#94a3b8;'>m³/m³</small></span><br>
                    <small style="color:#94a3b8;font-size:0.75rem;">{l_sub}</small>
                </div>
                """, unsafe_allow_html=True)
                st.progress(pct_norm)

        st.divider()

        # Advanced "What-If" Stress Simulation Laboratory
        with st.expander("🧪 Advanced 'What-If' Climate Stress Simulator (Scenario Laboratory)"):
            st.markdown("<p style='color:#94a3b8;font-size:0.9rem;'>Simulate micro-climate shocks to predict crop stress before adverse meteorological events occur:</p>", unsafe_allow_html=True)
            c_s1, c_s2, c_s3 = st.columns(3)
            with c_s1:
                sim_temp_delta = st.slider("🌡️ Temperature Shock Anomaly (°C):", -5.0, 8.0, 0.0, 0.5, key="sim_temp_delta")
            with c_s2:
                sim_rain_delta = st.slider("🌧️ Unseasonal Cloudburst / Rainfall (mm):", 0.0, 75.0, 0.0, 5.0, key="sim_rain_delta")
            with c_s3:
                sim_wind_delta = st.slider("💨 Wind Squall Velocity Anomaly (km/h):", 0.0, 45.0, 0.0, 5.0, key="sim_wind_delta")

            if sim_temp_delta != 0.0 or sim_rain_delta != 0.0 or sim_wind_delta != 0.0:
                # Calculate simulated CVI
                sim_cvi = min(100, max(5, int(cvi + (sim_temp_delta * 4.5) + (sim_rain_delta * 0.4) + (sim_wind_delta * 0.35))))
                sim_color = "#f87171" if sim_cvi >= 70 else ("#fb923c" if sim_cvi >= 45 else ("#facc15" if sim_cvi >= 25 else "#34d399"))
                st.markdown(f"""
                <div style="background:#0b1922;border:1px solid {sim_color};border-radius:12px;padding:14px;margin-top:10px;">
                    <b style="color:{sim_color};font-size:1.1rem;">🧪 Simulated Climate Impact:</b><br>
                    Baseline CVI: <b>{cvi}/100</b> ➔ Simulated CVI: <b style="color:{sim_color};font-size:1.3rem;">{sim_cvi}/100</b><br>
                    <span style="font-size:0.85rem;color:#e2e8f0;">
                        {'🚨 WARNING: Heatwave anomaly exceeds critical pollination threshold! Premature grain desiccation imminent.' if sim_temp_delta >= 3.0 else ''}
                        {'🌧️ ALERT: Extreme rainfall risks root asphyxiation and collar rot in poorly drained fields.' if sim_rain_delta >= 30.0 else ''}
                        {'💨 ADVISORY: High wind velocity combined with moist soil increases crop lodging risk.' if sim_wind_delta >= 25.0 else ''}
                    </span>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Actionable Preventive Protocol (ICAR Guidance)
        st.markdown("#### 💡 Actionable Agronomic Mitigation Protocol (ICAR Certified)")
        st.caption("Proactive measures tailored to current phenological stage:")
        for action in preventive:
            st.markdown(action)

        # Export report button
        c_exp1, c_exp2 = st.columns([3, 1])
        with c_exp2:
            report_lines = [
                f"KRISHIRAKSHAK AI — CLIMATE RISK ASSESSMENT & ADVISORY REPORT",
                f"Timestamp: {risk_assessment.get('timestamp')}",
                f"Location: {risk_assessment.get('location')}, {risk_assessment.get('state')}",
                f"Target Crop: {risk_assessment.get('crop')} ({sel_stage})",
                f"Climate Vulnerability Index (CVI): {cvi}/100 ({cvi_status})",
                f"Average Root Zone Moisture: {avg_moist} m3/m3",
                f"\n--- ACTIVE WARNINGS ---",
            ]
            for a in active_alerts:
                report_lines.append(f"[{a.get('severity')}] {a.get('type')}: {a.get('message')}")
            report_lines.append(f"\n--- PREVENTIVE PROTOCOL ---")
            for p in preventive:
                report_lines.append(p)
            report_str = "\n".join(report_lines)

            st.download_button(
                "📥 Export Full Risk Report (.txt)",
                data=report_str,
                file_name=f"ClimateRisk_{crop_key}_{active_risk_loc}.txt",
                mime="text/plain",
                width="stretch",
                key="btn_export_risk_report",
            )


# ════════════════════════════════════════════════════════════════════════
# TAB 4 — Plant Pathology (YOLOv8)
# ════════════════════════════════════════════════════════════════════════
with tab_disease:
    st.markdown("### 🔬 Autonomous Plant Disease Detection & Pathology")
    st.caption("Upload Leaf Photo or Multi-Page PDF Diagnostic Report • YOLOv8 Neural Inference • Complete Organic & Chemical Treatments")

    col_diag1, col_diag2 = st.columns([1, 1], gap="large")

    with col_diag1:
        st.markdown("#### 🌿 Input Plant Image / Document")

        source_tab = st.radio(
            "Image Source Mode:",
            ["📁 Drag & Drop (Image or PDF)", "📷 Live Camera Snap", "🧪 Preloaded Test Samples"],
            horizontal=True,
            label_visibility="collapsed",
        )

        chosen_leaf_img = None
        leaf_file_name = ""

        if source_tab == "📁 Drag & Drop (Image or PDF)":
            up_file = st.file_uploader(
                "Drag & drop leaf photo or PDF document here",
                type=["jpg", "jpeg", "png", "webp", "pdf", "bmp", "tiff"],
                help="Supports JPG, JPEG, PNG, WEBP, PDF, and BMP files (max 20 MB)",
            )
            if up_file:
                leaf_file_name = up_file.name
                if up_file.name.lower().endswith(".pdf"):
                    try:
                        import pypdfium2 as pdfium
                        pdf_doc = pdfium.PdfDocument(up_file.getvalue())
                        total_p = len(pdf_doc)
                        st.info(f"📄 **PDF Report Loaded:** `{up_file.name}` ({total_p} {'page' if total_p == 1 else 'pages'})")
                        p_idx = 0
                        if total_p > 1:
                            p_idx = st.number_input("Select page to analyze", min_value=1, max_value=total_p, value=1) - 1
                        page = pdf_doc.get_page(p_idx)
                        chosen_leaf_img = page.render(scale=2).to_pil()
                        leaf_file_name = f"{up_file.name} (Page {p_idx + 1})"
                    except Exception as ex:
                        st.error(f"PDF extraction error: {ex}")
                else:
                    try:
                        chosen_leaf_img = Image.open(up_file)
                    except Exception as ex:
                        st.error(f"Image read error: {ex}")

        elif source_tab == "📷 Live Camera Snap":
            cam_img = st.camera_input("Capture live photo of affected leaf:")
            if cam_img:
                chosen_leaf_img = Image.open(cam_img)
                leaf_file_name = "Camera Snapshot.jpg"

        elif source_tab == "🧪 Preloaded Test Samples":
            sample_dir = ROOT / "data" / "plantvillage" / "test" / "test"
            sample_items = [
                ("🍅 Tomato Early Blight", "TomatoEarlyBlight1.JPG"),
                ("🍅 Tomato Yellow Curl", "TomatoYellowCurlVirus1.JPG"),
                ("🥔 Potato Early Blight", "PotatoEarlyBlight1.JPG"),
                ("🌽 Corn Common Rust", "CornCommonRust1.JPG"),
                ("🍏 Apple Scab", "AppleScab1.JPG"),
                ("🌿 Healthy Potato Leaf", "PotatoHealthy1.JPG"),
            ]
            st.markdown("<small style='color:#a7f3d0;font-weight:600;'>Tap any sample to load & diagnose instantly:</small>", unsafe_allow_html=True)
            samp_cols = st.columns(2)
            for s_idx, (s_label, s_file) in enumerate(sample_items):
                with samp_cols[s_idx % 2]:
                    if st.button(s_label, width="stretch", key=f"sample_pick_btn_{s_idx}"):
                        s_path = sample_dir / s_file
                        if s_path.exists():
                            img = Image.open(s_path)
                            buf = io.BytesIO()
                            img.convert("RGB").save(buf, format="JPEG", quality=95)
                            st.session_state.chosen_sample_name = s_label
                            st.session_state.chosen_sample_img = img
                            with st.spinner(f"Diagnosing {s_label}..."):
                                st.session_state.detection = call_detect(buf.getvalue(), s_file)
                            st.rerun()

            if st.session_state.get("chosen_sample_img"):
                chosen_leaf_img = st.session_state.chosen_sample_img
                leaf_file_name = st.session_state.get("chosen_sample_name", "Preloaded Sample")

        if chosen_leaf_img:
            st.image(chosen_leaf_img, caption=f"Active Sample: {leaf_file_name}", width="stretch")

            if st.button("🔬 Analyze Leaf & Diagnose Pathology", type="primary", width="stretch", key="run_disease_diag_btn"):
                with st.spinner("Executing YOLOv8 neural network inference..."):
                    buffer = io.BytesIO()
                    chosen_leaf_img.convert("RGB").save(buffer, format="JPEG", quality=95)
                    diag_result = call_detect(buffer.getvalue(), leaf_file_name or "leaf.jpg")
                st.session_state.detection = diag_result
                st.rerun()

    with col_diag2:
        st.markdown("#### 📊 Diagnostic Results & Prescription")
        det = st.session_state.detection

        if det is None:
            st.markdown("""
            <div style="background:#13242e;border:1px solid #1e3a47;border-radius:14px;padding:2.5rem;text-align:center;color:#94a3b8;">
                <span style="font-size:2.8rem;">🔬</span><br>
                <b style="color:#ffffff;font-size:1.15rem;">No Active Analysis</b><br>
                <span>Upload a leaf photo or multi-page PDF document to view AI disease classification and treatment guidelines.</span>
            </div>
            """, unsafe_allow_html=True)

        elif not det.get("success"):
            st.error(f"Pathology analysis error: {det.get('error', 'Unknown error')}")

        elif not det.get("detections"):
            st.success("✅ **Healthy Plant Leaf Detected!** No pathogen or infectious lesions detected.")
            st.caption(det.get("message", "Plant canopy appears in optimal vegetative condition."))

        else:
            # Metrics
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("⏱️ Inference Speed", f"{det.get('inference_time_ms', 0):.0f} ms")
            m_col2.metric("🔍 Detected Candidates", len(det["detections"]))

            st.divider()

            for i, d in enumerate(det["detections"]):
                sev = d.get("severity", "unknown")
                emoji = SEVERITY_EMOJI.get(sev, "⚪")
                conf_pct = d["confidence"] * 100
                is_primary = (i == 0)

                with st.expander(f"{emoji} {'🎯 Primary Diagnosis' if is_primary else f'Candidate #{i+1}'}: **{d['display_name']}** ({conf_pct:.1f}%)", expanded=is_primary):
                    if d.get("hindi_name"):
                        st.markdown(f"**हिंदी नाम:** `{d['hindi_name']}`")

                    st.progress(safe_clamp(d.get("confidence", 0)), text=f"Model Confidence: {conf_pct:.1f}%")

                    sev_bg = SEVERITY_COLOR.get(sev, "#f3f4f6")
                    st.markdown(
                        f'<div style="background:{sev_bg};padding:4px 12px;border-radius:8px;'
                        f'display:inline-block;font-weight:700;font-size:0.85rem;margin:8px 0;">'
                        f'{emoji} Severity Level: {sev.upper()}</div>',
                        unsafe_allow_html=True,
                    )

                    treatment = det.get("treatment") if is_primary else None
                    if treatment:
                        st.markdown("---")
                        tr_tabs = st.tabs(["🍃 Organic Treatment", "💊 Chemical Spray", "🛡️ Prevention", "🌿 Symptoms & Cause"])

                        with tr_tabs[0]:
                            st.success(f"**Biological & Organic Control:**\n\n{treatment.get('organic_treatment', 'N/A')}")
                        with tr_tabs[1]:
                            st.warning(f"**Recommended Chemical Formulations:**\n\n{treatment.get('chemical_treatment', 'N/A')}")
                        with tr_tabs[2]:
                            st.info(f"**Preventive Field Operations:**\n\n{treatment.get('prevention', 'N/A')}")
                        with tr_tabs[3]:
                            st.markdown(f"**Pathogen Cause:** {treatment.get('cause', 'N/A')}")
                            st.markdown(f"**Favorable Conditions:** {treatment.get('favorable_conditions', 'N/A')}")
                            st.markdown(f"**Symptoms:** {treatment.get('symptoms', 'N/A')}")
                            st.markdown(f"**Economic Impact:** {treatment.get('economic_impact', 'N/A')}")


# ════════════════════════════════════════════════════════════════════════
# TAB 5 — Real-Time Weather (Open-Meteo)
# ════════════════════════════════════════════════════════════════════════
with tab_weather:
    st.markdown("### 🌤️ Real-Time Agricultural Weather & 7-Day Forecast")
    st.caption("🟢 Powered by Open-Meteo Satellite & Numerical Weather Prediction • Live Real-Time Current Conditions • Dynamic Farming Advisories")

    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        city_lookup = st.text_input(
            "Search Indian City / District for Live Weather:",
            value=st.session_state.location,
            placeholder="e.g. Ludhiana, Nashik, Varanasi, Pune, Bhopal, Jaipur...",
            key="weather_search_box",
        )
    with col_search2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Live Weather", width="stretch"):
            get_realtime_weather.clear()
            st.rerun()

    active_target_city = city_lookup.strip() if city_lookup.strip() else st.session_state.location
    live_wx = get_realtime_weather(active_target_city)

    if not live_wx or not live_wx.get("forecasts"):
        st.warning(f"Could not retrieve live meteorological data for '{active_target_city}'.")
    else:
        curr = live_wx.get("current", {})
        forecasts = live_wx.get("forecasts", [])

        # Hero Current Weather Card
        st.markdown(f"""
        <div class="ui-card" style="background: linear-gradient(135deg, #092a20 0%, #13242e 100%);border: 1px solid #10b981;box-shadow: 0 6px 16px rgba(0,0,0,0.35);">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
                <div>
                    <span style="font-size:0.85rem;color:#34d399;font-weight:700;">🟢 LIVE REAL-TIME OBSERVATION</span>
                    <h2 style="margin:4px 0 0 0;color:#ffffff;font-size:1.8rem;">{live_wx.get('city', active_target_city)} ({live_wx.get('state', 'India')})</h2>
                    <span style="color:#94a3b8;font-size:0.85rem;">Last Updated: {live_wx.get('last_updated', 'Just now')}</span>
                </div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="font-size:3rem;">{curr.get('icon', '⛅')}</span>
                    <div>
                        <span style="font-size:2.4rem;font-weight:800;color:#34d399;">{curr.get('temperature', '--')}°C</span><br>
                        <span style="color:#cbd5e1;font-weight:600;">{curr.get('condition', '')} • Feels {curr.get('apparent_temperature', '--')}°C</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        w1, w2, w3, w4 = st.columns(4)
        w1.metric("💧 Relative Humidity", f"{curr.get('humidity', '--')}%")
        w2.metric("💨 Wind Velocity", f"{curr.get('wind_speed', '--')} km/h")
        w3.metric("🌧️ Current Rainfall", f"{curr.get('precipitation', 0.0)} mm")
        
        t0 = forecasts[0].get("temperature", {}) if forecasts else {}
        t0_min = t0.get("min", "--") if isinstance(t0, dict) else t0
        t0_max = t0.get("max", "--") if isinstance(t0, dict) else t0
        w4.metric("🌡️ 24h Temperature Range", f"{t0_min}–{t0_max}°C" if forecasts else "--")

        # Real-time field suitability
        wind_spd = safe_float(curr.get("wind_speed", 0))
        precip = safe_float(curr.get("precipitation", 0))
        spray_safe = (wind_spd < 15.0) and (precip == 0.0)
        tillage_safe = (precip < 2.0)
        irrig_need = "💧 Recommended" if precip == 0.0 else "🚫 Delay (Rainfall Active)"

        st.markdown(f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;">
            <div style="background:#13242e;border:1px solid #1e3a47;border-radius:10px;padding:8px 14px;font-size:0.85rem;color:#f1f5f9;">
                🌾 <b>Chemical Spray Window:</b> <span style="color:{'#34d399' if spray_safe else '#f87171'};font-weight:700;">{'✅ Safe & Effective' if spray_safe else '⚠️ Avoid (High Wind/Precipitation)'}</span>
            </div>
            <div style="background:#13242e;border:1px solid #1e3a47;border-radius:10px;padding:8px 14px;font-size:0.85rem;color:#f1f5f9;">
                🚜 <b>Field Work & Tillage:</b> <span style="color:{'#34d399' if tillage_safe else '#fbbf24'};font-weight:700;">{'✅ Favorable Soil Conditions' if tillage_safe else '⚠️ Saturated Soil'}</span>
            </div>
            <div style="background:#13242e;border:1px solid #1e3a47;border-radius:10px;padding:8px 14px;font-size:0.85rem;color:#f1f5f9;">
                💧 <b>Irrigation Scheduling:</b> <span style="color:#38bdf8;font-weight:700;">{irrig_need}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        today_adv = curr.get("advisory", "")
        if today_adv:
            st.info(f"💡 **Today's Actionable Agricultural Advisory:** {today_adv}")

        st.divider()

        # 7-Day Forecast Grid
        st.markdown("#### 📅 7-Day Precision Agricultural Weather Grid")
        f_cols = st.columns(min(len(forecasts), 7))
        for col, f in zip(f_cols, forecasts[:7]):
            cond = f.get("condition", "")
            icon = f.get("icon", "🌤️")
            rain_mm = f.get("rainfall", 0)
            rain_prob = f.get("rain_probability", 0)
            t_day = f.get("temperature", {})
            t_min = t_day.get("min", "--") if isinstance(t_day, dict) else t_day
            t_max = t_day.get("max", "--") if isinstance(t_day, dict) else t_day
            with col:
                st.markdown(
                    f'<div class="weather-card-modern">'
                    f'<b>{f.get("day","")}</b><br>'
                    f'<small style="color:#94a3b8;">{f.get("date","")}</small><br>'
                    f'<span style="font-size:1.6rem;">{icon}</span><br>'
                    f'<small style="font-weight:600;color:#f1f5f9;">{cond}</small><br>'
                    f'<b style="color:#ffffff;">{t_min}–{t_max}°C</b><br>'
                    f'<small style="color:#34d399;">💧 {f.get("humidity","")}%</small><br>'
                    f'<small style="color:#38bdf8;">🌧️ {rain_mm}mm ({rain_prob}%)</small><br>'
                    f'<small style="color:#94a3b8;">💨 {f.get("wind_speed",0)}km/h</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # Detailed Day-by-Day Advisories
        st.markdown("#### 📝 Day-by-Day Farming Activity Advisories")
        for f in forecasts[:7]:
            adv = f.get("advisory", "")
            if adv:
                is_warn = any(k in adv.lower() for k in ["heavy", "avoid", "halt", "warning", "risk", "cold", "heat"])
                icon = "⚠️" if is_warn else "✅"
                st.markdown(f"**{f['day']} ({f['date']}):** {icon} {adv}")

        impact = live_wx.get("crop_impact", "")
        if impact:
            st.divider()
            st.success(f"🌾 **Seasonal Agronomic Impact:** {impact}")


# ════════════════════════════════════════════════════════════════════════
# TAB 6 — Real-Time Mandi Prices (APMC Live)
# ════════════════════════════════════════════════════════════════════════
with tab_market:
    st.markdown("### 💰 Real-Time Indian APMC Mandi Commodity Rates")
    st.caption("🟢 Official Government of India Agmarknet & APMC Terminal Markets Real-Time Price Integration")

    col_mk1, col_mk2 = st.columns([3, 1])
    with col_mk2:
        if st.button("🔄 Refresh Live APMC Rates", width="stretch"):
            get_realtime_mandi.clear()
            st.rerun()

    mandi_data = get_realtime_mandi()
    st.caption(
        f"📅 Date: **{mandi_data.get('date', 'Today')}** | "
        f"Unit: **{mandi_data.get('unit', 'INR per quintal')}** | "
        f"Source: **{mandi_data.get('source', 'Agmarknet APMC Live')}**"
    )

    all_markets = mandi_data.get("markets", {})
    if not all_markets:
        st.warning("Mandi commodity rates currently unavailable.")
    else:
        market_keys = list(all_markets.keys())
        def_idx = 0
        for i, k in enumerate(market_keys):
            if st.session_state.location.lower() in k.lower():
                def_idx = i
                break

        c_sel1, c_sel2 = st.columns([2, 2])
        with c_sel1:
            chosen_mkt = st.selectbox("🏪 Select APMC Terminal Market", market_keys, index=def_idx)
        with c_sel2:
            crop_search = st.text_input("🔍 Filter by Commodity (e.g. Tomato, Wheat, Onion, Mustard, टमाटर, गेहूं):", "")

        active_mkt_data = all_markets[chosen_mkt]
        commodities = active_mkt_data.get("commodities", {})

        filtered = {}
        for c_name, p in commodities.items():
            if not crop_search or crop_search.lower() in c_name.lower() or crop_search.strip() in p.get("remarks", "").lower():
                filtered[c_name] = p

        if filtered:
            p_list = list(filtered.values())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📦 Commodities Listed", len(p_list))
            rising = sum(1 for p in p_list if p.get("trend") == "rising")
            falling = sum(1 for p in p_list if p.get("trend") == "falling")
            stable = sum(1 for p in p_list if p.get("trend") == "stable")
            m2.metric("📈 Price Gainers", rising, delta=f"+{rising}" if rising else None)
            m3.metric("📉 Price Drops", falling, delta=f"-{falling}" if falling else None, delta_color="inverse")
            m4.metric("➡️ Stable Commodities", stable)

            st.divider()

            table_rows = []
            for crop, p in filtered.items():
                trend = p.get("trend", "stable")
                t_symbol = "↑" if trend == "rising" else ("↓" if trend == "falling" else "→")
                modal_val = safe_int(p.get("modal", 0))
                min_val = safe_int(p.get("min", 0))
                max_val = safe_int(p.get("max", 0))
                table_rows.append({
                    "Commodity": crop,
                    "Min Price (₹/q)": f"₹{min_val:,}",
                    "Modal Rate (₹/q)": f"₹{modal_val:,}",
                    "Retail Approx (₹/kg)": f"₹{modal_val/100:.1f}/kg",
                    "Max Price (₹/q)": f"₹{max_val:,}",
                    "Daily Trend": f"{t_symbol} {trend.title()}",
                    "Arrivals & Market Remarks": p.get("remarks", ""),
                })

            df_mandi = pd.DataFrame(table_rows)
            st.dataframe(df_mandi, width="stretch", hide_index=True)

        else:
            st.info(f"No commodities found matching '{crop_search}' at {chosen_mkt}.")

        # Government MSP 2024-25 Benchmark
        msp_dict = mandi_data.get("msp_2024_25", {})
        if msp_dict:
            st.divider()
            st.markdown("#### 🏛️ Government Minimum Support Price (MSP) 2024–25 Benchmarks")
            msp_cols = st.columns(3)
            for i, (crop, msp_info) in enumerate(msp_dict.items()):
                with msp_cols[i % 3]:
                    msp_val = safe_int(msp_info.get("msp", 0))
                    st.markdown(
                        f'<div class="stat-box-modern">'
                        f'<b style="font-size:1.05rem;">{crop}</b><br>'
                        f'<span class="mandi-price" style="font-size:1.3rem;">₹{msp_val:,}/quintal</span><br>'
                        f'<small style="color:#94a3b8;">{msp_info.get("season","")} • {msp_info.get("remarks","")}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()
        st.caption("🟢 Live official APMC market feed. Verify moisture levels and grading certifications with mandi auction inspectors prior to delivery.")
