"""
Weather Tool — Provides real-time live weather forecasts and agricultural advisories
using Open-Meteo live API with graceful fallback to cached weather data.
Formats clean, concise, non-repetitive advisories for farmers.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import Tool
from services.weather_service import get_weather_service

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "weather_data.json"

_weather_data: Optional[dict] = None


def _load_weather_data() -> dict:
    global _weather_data
    if _weather_data is None:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            _weather_data = json.load(f)
    return _weather_data


HINDI_CITY_MAP = {
    "दिल्ली": "Delhi",
    "मुंबई": "Mumbai",
    "पुणे": "Pune",
    "लखनऊ": "Lucknow",
    "जयपुर": "Jaipur",
    "लुधियाना": "Ludhiana",
    "चंडीगढ़": "Chandigarh",
    "भोपाल": "Bhopal",
    "पटना": "Patna",
    "अमृतसर": "Amritsar",
    "बठिंडा": "Bathinda",
    "जालंधर": "Jalandhar",
    "पटियाला": "Patiala",
}


def get_weather(query: str, default_location: str = "Delhi") -> str:
    """
    Returns real-time live weather forecast and agronomic advisories for a location.
    Formatted cleanly without redundant text or duplicate lines.
    """
    try:
        # Determine location from query or default
        location = None
        for h_city, e_city in HINDI_CITY_MAP.items():
            if h_city in query:
                location = e_city
                break
        if not location:
            for city in ["Delhi", "Mumbai", "Pune", "Lucknow", "Jaipur", "Ludhiana", "Chandigarh", "Bhopal", "Patna", "Amritsar", "Bathinda", "Jalandhar", "Patiala"]:
                if city.lower() in query.lower():
                    location = city
                    break
        if not location:
            location = default_location or "Delhi"

        # Query live WeatherService
        svc = get_weather_service()
        live_wx = svc.get_live_weather(location)

        city_name = live_wx.get("city", location)
        state_name = live_wx.get("state", "")
        curr = live_wx.get("current", {})
        forecasts = live_wx.get("forecasts", [])
        crop_impact = live_wx.get("crop_impact", "")

        lines = [
            f"🌤 **मौसम पूर्वानुमान — {city_name} ({state_name})**",
            f"• 🌡️ **वर्तमान तापमान**: **{curr.get('temperature', '--')}°C** | {curr.get('condition', '')} {curr.get('icon', '')}",
            f"• 💧 **नमी (Humidity)**: {curr.get('humidity', '--')}% | 💨 **हवा**: {curr.get('wind_speed', '--')} km/h",
            f"• 🌧️ **बारिश की संभावना**: {curr.get('precipitation', 0.0)} mm",
            f"• 💡 **आज की कृषि सलाह**: {curr.get('advisory', 'अनुकूल मौसम। सामान्य सिंचाई एवं नियमित खरपतवार नियंत्रण करें।')}",
            "",
            "**अगले 3 दिनों का पूर्वानुमान:**",
        ]

        for f in forecasts[:3]:
            rain_val = f.get('rainfall', 0)
            rain_prob = f.get('rain_probability', 0)
            lines.append(
                f"• **{f['day']}**: {f['condition']} {f.get('icon', '')} | "
                f"तापमान: {f['temperature']['min']}–{f['temperature']['max']}°C | "
                f"बारिश: {rain_val} mm ({rain_prob}%)"
            )

        if crop_impact:
            lines.append(f"\n🌾 **फसल प्रभाव**: {crop_impact}")

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Live weather service notice ({e}), falling back to offline data.")
        try:
            data = _load_weather_data()
            matched_city = default_location or "Delhi"
            for city in data.get("cities", {}):
                if city.lower() in query.lower():
                    matched_city = city
                    break
            city_data = data["cities"].get(matched_city, {})
            curr = city_data.get("current", {})
            forecasts = city_data.get("forecasts", [])

            lines = [
                f"🌤 **मौसम बुलेटिन — {matched_city}**",
                f"• 🌡️ **तापमान**: {curr.get('temperature', 22)}°C | {curr.get('condition', 'साफ मौसम')}",
                f"• 💡 **सलाह**: अनुकूल परिस्थितियां। सामान्य कृषि कार्य जारी रखें।",
                "",
                "**आगामी पूर्वानुमान:**"
            ]
            for f in forecasts[:3]:
                lines.append(
                    f"• **{f['day']}**: {f['condition']} | "
                    f"{f['temperature']['min']}–{f['temperature']['max']}°C | "
                    f"बारिश: {f.get('rainfall', 0)} mm"
                )
            return "\n".join(lines)
        except Exception as ex:
            return f"मौसम जानकारी प्राप्त करने में असमर्थ: {str(ex)}"


def get_weather_tool(default_location: str = "Delhi") -> Tool:
    """Return a LangChain Tool for weather queries."""
    return Tool(
        name="weather_forecast",
        description=(
            "Get current weather forecast, rainfall probability, and field advisories. "
            "Use for questions about rain, temperature, wind, and irrigation windows."
        ),
        func=lambda q: get_weather(q, default_location=default_location),
    )
