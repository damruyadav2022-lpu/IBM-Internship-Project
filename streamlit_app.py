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

@st.cache_data(ttl=30)
def is_backend_online() -> bool:
    """Detect whether a dedicated FastAPI gateway is active on BACKEND_URL."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.2)
        return r.status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=300)
def load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=300)
def get_realtime_weather(location: str) -> dict:
    """Fetch real-time weather from backend API or local WeatherService."""
    if is_backend_online():
        try:
            r = requests.get(f"{API_URL}/weather", params={"location": location}, timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    try:
        sys.path.insert(0, str(ROOT / "backend"))
        from services.weather_service import get_weather_service
        return get_weather_service().get_live_weather(location)
    except Exception:
        return {}

@st.cache_data(ttl=300)
def get_realtime_mandi() -> dict:
    """Fetch real-time Mandi APMC prices from backend API or local MandiService."""
    if is_backend_online():
        try:
            r = requests.get(f"{API_URL}/mandi-prices", timeout=8)
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

@st.cache_data(ttl=300)
def get_climate_risk(location: str, crop: str = "wheat", stage: str = None) -> dict:
    """Fetch real-time AI Climate Risk and Early Warnings."""
    if is_backend_online():
        try:
            params = {"location": location, "crop": crop}
            if stage:
                params["stage"] = stage
            r = requests.get(f"{API_URL}/climate-risk", params=params, timeout=10)
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
                timeout=120,
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

def call_detect(image_bytes: bytes, filename: str) -> dict:
    if is_backend_online():
        try:
            r = requests.post(
                f"{API_URL}/detect-disease",
                files={"file": (filename, image_bytes, "image/jpeg")},
                params={"top_k": 3},
                timeout=60,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return _local_detect(image_bytes)

def _local_detect(image_bytes: bytes) -> dict:
    try:
        sys.path.insert(0, str(ROOT / "backend"))
        from disease_detector import get_detector
        detector = get_detector()
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
            "🌾 What crop is best for this rabi season?",
            "🌧️ What is today's farming weather advisory?",
            "💰 What is today's mandi rate for tomatoes?",
            "🐛 How to control yellow rust on wheat crop?",
            "🌱 What fertilizer schedule is best for mustard?",
            "🔍 How to identify late blight in potato?",
        ],
        "Hindi": [
            "🌾 इस रबी मौसम में कौन सी फसल सबसे अच्छी है?",
            "🌧️ आज की मौसम और कृषि सलाह क्या है?",
            "💰 आज दिल्ली और मुंबई में टमाटर का मंडी भाव क्या है?",
            "🐛 गेहूं में पीला रतुआ कीट से बचाव कैसे करें?",
            "🌱 सरसों के लिए संतुलित खाद और सिंचाई क्या है?",
            "🔍 आलू में पछेती झुलसा रोग की रोकथाम कैसे करें?",
        ],
        "Punjabi": [
            "🌾 ਇਸ ਹਾੜ੍ਹੀ ਰੁੱਤ ਵਿੱਚ ਕਿਹੜੀ ਫਸਲ ਸਭ ਤੋਂ ਵਧੀਆ ਹੈ?",
            "🌧️ ਅੱਜ ਦੀ ਮੌਸਮ ਅਤੇ ਖੇਤੀਬਾੜੀ ਸਲਾਹ ਕੀ ਹੈ?",
            "💰 ਅੱਜ ਮੰਡੀ ਵਿੱਚ ਟਮਾਟਰ ਅਤੇ ਆਲੂ ਦਾ ਰੇਟ ਕੀ ਹੈ?",
            "🐛 ਕਣਕ ਦੀ ਫਸਲ ਵਿੱਚ ਪੀਲੀ ਕੁੰਗੀ ਦੀ ਰੋਕਥਾਮ ਕਿਵੇਂ ਕਰੀਏ?",
            "🌱 ਸਰ੍ਹੋਂ ਦੀ ਫਸਲ ਲਈ ਖਾਦ ਅਤੇ ਪਾਣੀ ਦਾ ਪ੍ਰਬੰਧ ਕੀ ਹੈ?",
            "🔍 ਟਮਾਟਰ ਦੇ ਪੱਤਿਆਂ ਤੇ ਪੀਲੇ ਧੱਬਿਆਂ ਦਾ ਇਲਾਜ ਕੀ ਹੈ?",
        ],
    }

    cols = st.columns(3)
    active_questions = quick_qs.get(st.session_state.language, quick_qs["English"])
    for i, q in enumerate(active_questions):
        with cols[i % 3]:
            if st.button(q, width="stretch", key=f"quick_btn_{i}"):
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
                <b style="font-size:1.2rem;color:#34d399;">Welcome to KrishiRakshak AI</b><br>
                <span style="color:#cbd5e1;">Ask about crop calendars, pest management, live weather forecasts, or APMC mandi rates.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-bubble-user">🧑‍🌾 <b>Farmer ({st.session_state.location}):</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    sources_html = "".join([f'<span class="source-tag">📌 {s}</span>' for s in msg.get("sources", [])])
                    st.markdown(
                        f'<div class="chat-bubble-assistant">'
                        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                        f'<span style="font-size:1.2rem;">🌾</span><b style="color:#34d399;">KrishiRakshak AI:</b>'
                        f'</div>'
                        f'{msg["content"]}'
                        f'{"<div style=\'margin-top:12px;\'>" + sources_html + "</div>" if sources_html else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    steps = msg.get("steps", [])
                    if steps:
                        with st.expander(f"🧠 Agricultural Agent Reasoning ({len(steps)} Domain Tools Used)"):
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

    c_c1, c_c2 = st.columns([4, 1])
    with c_c2:
        if st.button("🗑️ Clear Conversation", width="stretch"):
            st.session_state.messages = []
            st.rerun()


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — AI Voice Assistant
# ════════════════════════════════════════════════════════════════════════
with tab_voice:
    st.markdown("### 🎙️ Multilingual AI Voice Farming Assistant")
    st.caption("ਬੋਲ ਕੇ ਖੇਤੀ ਸਲਾਹ ਲਵੋ • बोलकर खेती सलाह पाएं • Designed for Farmers Who Prefer Speaking Over Typing")

    col_v1, col_v2 = st.columns([1, 1], gap="large")

    with col_v1:
        st.markdown("""
        <div class="ui-card">
            <h4 style="margin-top:0;color:#ffffff;">🗣️ Spoken Voice Input</h4>
            <p style="color:#94a3b8;font-size:0.9rem;">Speak your farming queries naturally. Whisper AI transcribes your spoken language, routes it through our agronomy reasoning agent, and synthesizes human-like voice responses.</p>
        </div>
        """, unsafe_allow_html=True)

        # Voice language selection
        v_lang_sel = st.radio(
            "Select Voice Language / ਬੋਲੀ / भाषा:",
            ["Hindi (हिंदी)", "Punjabi (ਪੰਜਾਬੀ)", "English"],
            horizontal=True,
            key="v_lang_sel_radio",
        )
        clean_vlang = "Hindi" if "Hindi" in v_lang_sel else ("Punjabi" if "Punjabi" in v_lang_sel else "English")

        input_voice_mode = st.radio(
            "Input Method:",
            ["🎙️ Live Microphone", "📁 Upload Audio File", "💡 Spoken Farmer Presets"],
            horizontal=True,
        )

        audio_bytes_to_process = None
        preset_spoken_text = None

        if input_voice_mode == "🎙️ Live Microphone":
            mic_audio = st.audio_input(f"Tap microphone to speak in {clean_vlang}:")
            if mic_audio:
                audio_bytes_to_process = mic_audio.getvalue()
                st.audio(mic_audio, format="audio/wav")

        elif input_voice_mode == "📁 Upload Audio File":
            uploaded_audio = st.file_uploader(
                "Upload a voice note recording:",
                type=["wav", "mp3", "m4a", "webm", "ogg"],
                help="Supports WAV, MP3, M4A, WebM, and OGG formats",
            )
            if uploaded_audio:
                audio_bytes_to_process = uploaded_audio.getvalue()
                st.audio(uploaded_audio)

        elif input_voice_mode == "💡 Spoken Farmer Presets":
            st.markdown("**Tap any preset spoken farmer query to test:**")
            if clean_vlang == "Hindi":
                presets = [
                    "🍅 मेरे टमाटर के पत्तों पर पीले धब्बे आ रहे हैं, इसका इलाज क्या है?",
                    "💰 आज दिल्ली और आज़ादपुर मंडी में टमाटर का क्या भाव है?",
                    "🌧️ क्या आज गेहूं की फसल में कीटनाशक का छिड़काव कर सकते हैं?",
                    "🌾 रबी मौसम के लिए गेहूं की कौन सी किस्म सबसे ज्यादा पैदावार देती है?",
                ]
            elif clean_vlang == "Punjabi":
                presets = [
                    "🍅 ਮੇਰੇ ਟਮਾਟਰ ਦੇ ਪੱਤਿਆਂ ਤੇ ਪੀਲੇ ਧੱਬੇ ਆ ਰਹੇ ਹਨ, ਕੀ ਕਰੀਏ?",
                    "🌾 ਕਣਕ ਦੀ ਫਸਲ ਵਿੱਚ ਪੀਲੀ ਕੁੰਗੀ ਦੀ ਰੋਕਥਾਮ ਕਿਵੇਂ ਕਰੀਏ?",
                    "💰 ਅੱਜ ਮੰਡੀ ਵਿੱਚ ਟਮਾਟਰ ਅਤੇ ਆਲੂ ਦਾ ਰੇਟ ਕੀ ਹੈ?",
                    "🌧️ ਕੀ ਅੱਜ ਖੇਤ ਵਿੱਚ ਸਪਰੇਅ ਕਰਨਾ ਸਹੀ ਰਹੇਗਾ?",
                ]
            else:
                presets = [
                    "🍅 My tomato plant leaves have yellow spots and curling, what should I do?",
                    "💰 What is today's mandi market rate for tomatoes in Delhi?",
                    "🌧️ What is today's agricultural weather advisory for spraying?",
                    "🌾 How to effectively manage weed and nitrogen schedule for wheat?",
                ]

            for s in presets:
                if st.button(s, width="stretch", key=f"voice_preset_{s[:18]}"):
                    preset_spoken_text = s.split(" ", 1)[-1]

        process_voice = st.button("🚀 Process Voice Advice", type="primary", width="stretch")

    with col_v2:
        st.markdown("#### 🎧 Spoken Diagnosis & Voice Output")
        voice_output_container = st.container()

        if process_voice or preset_spoken_text:
            recognized_text = None

            if audio_bytes_to_process:
                with st.spinner("🎙️ Transcribing farmer voice with Whisper AI..."):
                    stt = call_transcribe(audio_bytes_to_process, language=clean_vlang)
                    if stt.get("success"):
                        recognized_text = stt.get("text", "")
                    else:
                        st.error(f"Transcription error: {stt.get('error')}")
            elif preset_spoken_text:
                recognized_text = preset_spoken_text

            if recognized_text:
                with voice_output_container:
                    # Farmer Query Box
                    st.markdown(
                        f'<div style="background:#0d231b;border:1px solid #10b981;border-radius:14px;padding:1rem;margin-bottom:1rem;color:#f1f5f9;">'
                        f'<b>🗣️ Farmer Spoken Query ({clean_vlang}):</b><br>'
                        f'<span style="font-size:1.15rem;color:#34d399;font-weight:700;">"{recognized_text}"</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    with st.spinner(f"🧠 KrishiRakshak diagnosing & retrieving remedies in {clean_vlang}..."):
                        agent_res = call_chat(recognized_text, clean_vlang, st.session_state.location)

                    resp_text = agent_res.get("response", "")
                    sources = agent_res.get("sources", [])

                    with st.spinner(f"🔊 Generating natural spoken {clean_vlang} audio..."):
                        speech_bytes = call_tts(resp_text, clean_vlang)

                    if speech_bytes:
                        st.markdown("##### 🔊 Spoken Audio Prescription")
                        st.audio(speech_bytes, format="audio/mp3", autoplay=True)

                    st.markdown("##### 📋 Prescribed Agricultural Action")
                    st.markdown(resp_text)

                    if sources:
                        sources_tags = " ".join([f'<span class="source-tag">📌 {s}</span>' for s in sources])
                        st.markdown(f"<div style='margin-top:10px;'><b>Verified Sources:</b><br>{sources_tags}</div>", unsafe_allow_html=True)

                    st.session_state.voice_chat_history.append({
                        "query": recognized_text,
                        "response": resp_text,
                        "audio": speech_bytes,
                        "lang": clean_vlang,
                    })

        elif st.session_state.voice_chat_history:
            last = st.session_state.voice_chat_history[-1]
            with voice_output_container:
                st.markdown(
                    f'<div style="background:#0d231b;border:1px solid #10b981;border-radius:14px;padding:1rem;margin-bottom:1rem;color:#f1f5f9;">'
                    f'<b>🗣️ Farmer Spoken Query ({last["lang"]}):</b><br>'
                    f'<span style="font-size:1.15rem;color:#34d399;font-weight:700;">"{last["query"]}"</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if last.get("audio"):
                    st.markdown("##### 🔊 Spoken Audio Prescription")
                    st.audio(last["audio"], format="audio/mp3")
                st.markdown("##### 📋 Prescribed Agricultural Action")
                st.markdown(last["response"])
        else:
            with voice_output_container:
                st.markdown("""
                <div style="background:#13242e;border:1px solid #1e3a47;border-radius:14px;padding:2.5rem;text-align:center;color:#94a3b8;">
                    <span style="font-size:2.5rem;">🎙️</span><br>
                    <b style="color:#ffffff;font-size:1.1rem;">Awaiting Spoken Query</b><br>
                    <span>Tap the microphone or choose a preset query to hear natural spoken farming advice.</span>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — AI Climate Risk Predictor & Early Warning System
# ════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown("### 🛡️ AI Climate Risk Predictor & Early Warning System")
    st.caption("🟢 Satellite-Derived Root-Zone Soil Moisture • 14-Day Hydro-Meteorological Intelligence • ICAR Agronomic Stress Guidelines")

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

    col_r1, col_r2, col_r3, col_r4 = st.columns([2, 2, 2, 1])
    with col_r1:
        sel_crop_name = st.selectbox("🌾 Target Crop:", list(crop_stages_options.keys()), key="risk_crop_select")
    crop_key, stage_choices = crop_stages_options[sel_crop_name]

    with col_r2:
        sel_stage = st.selectbox("🌱 Current Phenological Stage:", stage_choices, index=2, key="risk_stage_select")

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
        cvi = risk_assessment.get("climate_vulnerability_index", 0)
        cvi_status = risk_assessment.get("cvi_status", "NORMAL")
        cvi_icon = risk_assessment.get("cvi_icon", "🟢")
        sm = risk_assessment.get("satellite_soil_moisture", {})
        active_alerts = risk_assessment.get("active_alerts", [])
        scores = risk_assessment.get("risk_scores", {})
        preventive = risk_assessment.get("preventive_actions", [])

        # Midnight Forest palette for CVI
        if cvi >= 70:
            cvi_color = "#f87171"
            cvi_bg = "linear-gradient(135deg, #381216 0%, #1c0e11 100%)"
            cvi_border = "#ef4444"
        elif cvi >= 45:
            cvi_color = "#fb923c"
            cvi_bg = "linear-gradient(135deg, #331d0d 0%, #1c140d 100%)"
            cvi_border = "#f97316"
        elif cvi >= 25:
            cvi_color = "#facc15"
            cvi_bg = "linear-gradient(135deg, #2e240a 0%, #171c19 100%)"
            cvi_border = "#eab308"
        else:
            cvi_color = "#34d399"
            cvi_bg = "linear-gradient(135deg, #092a20 0%, #13242e 100%)"
            cvi_border = "#10b981"

        # Hero CVI Banner
        st.markdown(f"""
        <div style="background:{cvi_bg};border:2px solid {cvi_border};border-radius:16px;padding:1.25rem 1.75rem;margin-bottom:1.25rem;box-shadow:0 6px 16px rgba(0,0,0,0.35);">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                <div>
                    <span style="font-size:0.85rem;color:{cvi_color};font-weight:700;">🟢 REAL-TIME SATELLITE CLIMATE RISK ASSESSMENT</span>
                    <h2 style="margin:4px 0 0 0;color:#ffffff;font-size:1.6rem;">{risk_assessment.get('crop')} at {risk_assessment.get('location')} ({risk_assessment.get('state')})</h2>
                    <span style="color:#94a3b8;font-size:0.88rem;">Phenological Stage: <b style="color:#e2e8f0;">{sel_stage}</b> • Last Updated: {risk_assessment.get('timestamp')}</span>
                </div>
                <div style="display:flex;align-items:center;gap:16px;">
                    <div style="text-align:right;">
                        <span style="font-size:0.85rem;color:#94a3b8;font-weight:600;">Climate Vulnerability</span><br>
                        <span style="font-size:2.2rem;font-weight:800;color:{cvi_color};">{cvi}/100</span>
                    </div>
                    <div style="font-size:2.4rem;">{cvi_icon}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Active Alerts Section
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
            st.success(f"✅ **No Critical Climate Threats Detected**: Current weather conditions are favorable for {sel_crop_name} during {sel_stage}.")

        st.divider()

        # 6 Threat Cards
        st.markdown("#### 📊 Six-Vector Climate Risk Diagnostic")
        rk_col1, rk_col2, rk_col3 = st.columns(3)

        drought_info = scores.get("drought_risk", {})
        with rk_col1:
            st.markdown(
                f'<div class="ui-card">'
                f'<b>{drought_info.get("icon", "🏜️")} Drought & Moisture Stress</b><br>'
                f'<span style="font-size:1.4rem;font-weight:800;">{drought_info.get("score", 0)}%</span> '
                f'<small style="color:#94a3b8;">({drought_info.get("level", "LOW")})</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(safe_clamp(drought_info.get("score", 0) / 100.0))

        flood_info = scores.get("flood_risk", {})
        with rk_col2:
            st.markdown(
                f'<div class="ui-card">'
                f'<b>{flood_info.get("icon", "🌊")} Flood & Waterlogging Hazard</b><br>'
                f'<span style="font-size:1.4rem;font-weight:800;">{flood_info.get("score", 0)}%</span> '
                f'<small style="color:#94a3b8;">({flood_info.get("level", "LOW")})</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(safe_clamp(flood_info.get("score", 0) / 100.0))

        heat_info = scores.get("heat_stress", {})
        with rk_col3:
            st.markdown(
                f'<div class="ui-card">'
                f'<b>{heat_info.get("icon", "🌡️")} Terminal Heat Stress</b><br>'
                f'<span style="font-size:1.4rem;font-weight:800;">{heat_info.get("score", 0)}%</span> '
                f'<small style="color:#94a3b8;">({heat_info.get("level", "LOW")})</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(safe_clamp(heat_info.get("score", 0) / 100.0))

        rk_col4, rk_col5, rk_col6 = st.columns(3)

        frost_info = scores.get("frost_warning", {})
        with rk_col4:
            st.markdown(
                f'<div class="ui-card">'
                f'<b>{frost_info.get("icon", "❄️")} Frost & Radiative Chilling</b><br>'
                f'<span style="font-size:1.4rem;font-weight:800;">{frost_info.get("score", 0)}%</span> '
                f'<small style="color:#94a3b8;">({frost_info.get("level", "LOW")})</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(safe_clamp(frost_info.get("score", 0) / 100.0))

        lodging_info = scores.get("lodging_risk", {})
        with rk_col5:
            st.markdown(
                f'<div class="ui-card">'
                f'<b>{lodging_info.get("icon", "🌧️")} Heavy Rain & Lodging Hazard</b><br>'
                f'<span style="font-size:1.4rem;font-weight:800;">{lodging_info.get("score", 0)}%</span> '
                f'<small style="color:#94a3b8;">({lodging_info.get("level", "LOW")})</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(safe_clamp(lodging_info.get("score", 0) / 100.0))

        fungal_info = scores.get("fungal_disease_risk", {})
        with rk_col6:
            st.markdown(
                f'<div class="ui-card">'
                f'<b>{fungal_info.get("icon", "🦠")} Pathogen / Fungal Epidemic Risk</b><br>'
                f'<span style="font-size:1.4rem;font-weight:800;">{fungal_info.get("score", 0)}%</span> '
                f'<small style="color:#94a3b8;">({fungal_info.get("level", "LOW")})</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(safe_clamp(fungal_info.get("score", 0) / 100.0))

        st.divider()

        # Satellite Root Zone Soil Moisture Profile
        st.markdown("#### 🛰️ Real-Time Satellite Root-Zone Soil Moisture Profile (ECMWF ERA5-Land)")
        st.caption(f"Average Root Zone Moisture: **{sm.get('average_root_moisture')} m³/m³** • Status: **{sm.get('status')}** • Permanent Wilting Point: **{sm.get('wilting_threshold')} m³/m³**")

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("🌱 Surface (0–1 cm)", f"{sm.get('surface_0_to_1cm', '--')} m³/m³", "Topsoil Evaporation")
        sm2.metric("🌾 Seed Zone (1–3 cm)", f"{sm.get('seed_zone_1_to_3cm', '--')} m³/m³", "Germination Zone")
        sm3.metric("🌿 Active Roots (3–9 cm)", f"{sm.get('active_root_3_to_9cm', '--')} m³/m³", "Nutrient Uptake")
        sm4.metric("🌳 Subsoil (9–27 cm)", f"{sm.get('deep_root_9_to_27cm', '--')} m³/m³", "Storage Reserve")

        st.divider()

        # Preventive Agronomic Mitigation Protocol
        st.markdown("#### 💡 Actionable Preventive Protocol (ICAR Guidance)")
        st.caption("Proactive measures to protect crop health before environmental damage occurs:")
        for action in preventive:
            st.markdown(action)

        st.divider()

        # 14-Day Timeline (Historical Actuals + Forecast)
        timeline = risk_assessment.get("timeline", [])
        if timeline:
            with st.expander("📈 14-Day Hydro-Meteorological Observation Timeline (Past 7 Days + Next 7 Days)"):
                df_time = pd.DataFrame(timeline)
                df_time.rename(columns={
                    "date": "Date",
                    "type": "Observation Phase",
                    "max_temp": "Max Temp (°C)",
                    "min_temp": "Min Temp (°C)",
                    "rainfall": "Rainfall (mm)",
                }, inplace=True)
                st.dataframe(df_time, width="stretch", hide_index=True)


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
            sample_options = {
                "🍅 Tomato Early Blight": "TomatoEarlyBlight1.JPG",
                "🍅 Tomato Yellow Leaf Curl": "TomatoYellowCurlVirus1.JPG",
                "🍅 Healthy Tomato Leaf": "TomatoHealthy1.JPG",
                "🥔 Potato Early Blight": "PotatoEarlyBlight1.JPG",
                "🥔 Healthy Potato Leaf": "PotatoHealthy1.JPG",
                "🌽 Corn Common Rust": "CornCommonRust1.JPG",
                "🍏 Apple Scab Disease": "AppleScab1.JPG",
            }
            sample_choice = st.selectbox("Select high-resolution sample leaf:", list(sample_options.keys()))
            sample_file = sample_options[sample_choice]
            sample_path = sample_dir / sample_file
            if sample_path.exists():
                chosen_leaf_img = Image.open(sample_path)
                leaf_file_name = f"Sample: {sample_choice}"
            else:
                st.warning("Sample image not found on disk.")

        if chosen_leaf_img:
            st.image(chosen_leaf_img, caption=f"Active Sample: {leaf_file_name}", width="stretch")

            if st.button("🔬 Analyze Leaf & Diagnose Pathology", type="primary", width="stretch"):
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
