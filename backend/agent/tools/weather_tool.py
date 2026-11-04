"""
Weather Tool — reads weather_data.json and returns forecasts for a given location.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import Tool

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
}


def _find_city(location: str, data: dict) -> Optional[str]:
    """Case-insensitive partial match for city name, supporting Hindi names."""
    location_lower = location.lower().strip()
    for h_city, e_city in HINDI_CITY_MAP.items():
        if h_city in location or h_city.lower() in location_lower:
            return e_city
    for city in data.get("cities", {}):
        if location_lower in city.lower() or city.lower() in location_lower:
            return city
    return None


"""
Weather Tool — Provides real-time live weather forecasts and agricultural advisories
using Open-Meteo live API with graceful fallback to cached weather data.
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
}


def get_weather(query: str, default_location: str = "Delhi") -> str:
    """
    Returns real-time live weather forecast and agronomic advisories for a location.
    Query can be a city name or a question about weather.
    """
    try:
        # Determine location from query or default
        location = None
        for h_city, e_city in HINDI_CITY_MAP.items():
            if h_city in query:
                location = e_city
                break
        if not location:
            for city in ["Delhi", "Mumbai", "Pune", "Lucknow", "Jaipur", "Ludhiana", "Chandigarh", "Bhopal", "Patna"]:
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
        updated = live_wx.get("last_updated", "")

        lines = [
            f"🌤 Real-Time Live Weather Forecast for {city_name} ({state_name}):",
            f"🟢 Source: Open-Meteo Real-Time Weather | Updated: {updated}",
            f"🌡️ Current Condition: {curr.get('temperature', '--')}°C | {curr.get('condition', '')} {curr.get('icon', '')}",
            f"💧 Humidity: {curr.get('humidity', '--')}% | 💨 Wind: {curr.get('wind_speed', '--')} km/h | 🌧️ Precipitation: {curr.get('precipitation', 0.0)} mm",
            f"💡 Today's Farm Advisory: {curr.get('advisory', '')}",
            "",
            "📅 Next 3-Day Forecast:",
        ]

        for f in forecasts[:3]:
            rain_info = f"Rain: {f.get('rainfall', 0)} mm ({f.get('rain_probability', 0)}% prob)"
            lines.append(
                f"• {f['day']} ({f['date']}): {f['condition']} {f.get('icon', '')} | "
                f"Temp: {f['temperature']['min']}–{f['temperature']['max']}°C | "
                f"Humidity: {f['humidity']}% | {rain_info}"
            )
            lines.append(f"  Advisory: {f['advisory']}")

        if crop_impact:
            lines.append("")
            lines.append(f"🌾 Crop Impact: {crop_impact}")

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Live weather service notice ({e}), falling back to offline data.")
        try:
            data = _load_weather_data()
            matched_city = "Delhi"
            for city in data.get("cities", {}):
                if city.lower() in query.lower():
                    matched_city = city
                    break
            city_data = data["cities"].get(matched_city, {})
            forecasts = city_data.get("forecasts", [])
            lines = [
                f"🌤 Weather Forecast for {matched_city} ({city_data.get('state', '')}):",
                f"Last Updated: {data.get('last_updated', 'N/A')}",
                "",
            ]
            for f in forecasts[:3]:
                lines.append(
                    f"📅 {f['day']} ({f['date']}): {f['condition']} | "
                    f"Temp: {f['temperature']['min']}–{f['temperature']['max']}°C | "
                    f"Humidity: {f['humidity']}% | Rainfall: {f['rainfall']} mm"
                )
                lines.append(f"   💡 Advisory: {f['advisory']}")
            return "\n".join(lines)
        except Exception as ex:
            return f"Error retrieving weather data: {str(ex)}"



def get_weather_tool(default_location: str = "Delhi") -> Tool:
    """Return a LangChain Tool for weather queries."""
    return Tool(
        name="weather_forecast",
        description=(
            "Get current weather forecast and farming advisories for Indian cities. "
            "Provide a city name (e.g., Delhi, Mumbai, Pune, Lucknow, Jaipur) "
            "or ask about weather conditions. "
            "Use this to advise farmers on irrigation, spraying, and field activities."
        ),
        func=lambda query: get_weather(query, default_location),
    )
