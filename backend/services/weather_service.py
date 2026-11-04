"""
Weather Service — Fetches real-time live weather and 7-day agricultural forecasts
using the Open-Meteo API (free, open-access, zero API key required).
Provides live geocoding, real-time current conditions, and dynamic agronomic advisories.
"""
import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Predefined coordinates for common Indian agricultural hubs
MAJOR_CITIES: Dict[str, Tuple[float, float, str]] = {
    "delhi": (28.6139, 77.2090, "Delhi"),
    "mumbai": (19.0760, 72.8777, "Maharashtra"),
    "pune": (18.5204, 73.8567, "Maharashtra"),
    "lucknow": (26.8467, 80.9462, "Uttar Pradesh"),
    "jaipur": (26.9124, 75.7873, "Rajasthan"),
    "ludhiana": (30.9010, 75.8573, "Punjab"),
    "chandigarh": (30.7333, 76.7794, "Punjab/Haryana"),
    "bhopal": (23.2599, 77.4126, "Madhya Pradesh"),
    "patna": (25.5941, 85.1376, "Bihar"),
    "ahmedabad": (23.0225, 72.5714, "Gujarat"),
    "hyderabad": (17.3850, 78.4867, "Telangana"),
    "bengaluru": (12.9716, 77.5946, "Karnataka"),
    "kolkata": (22.5726, 88.3639, "West Bengal"),
    "varanasi": (25.3176, 82.9739, "Uttar Pradesh"),
    "indore": (22.7196, 75.8577, "Madhya Pradesh"),
    "nagpur": (21.1458, 79.0882, "Maharashtra"),
    "nashik": (19.9975, 73.7898, "Maharashtra"),
    "amritsar": (31.6340, 74.8723, "Punjab"),
}

# Multilingual city name mapping
CITY_TRANSLATIONS: Dict[str, str] = {
    "दिल्ली": "delhi",
    "मुंबई": "mumbai",
    "पुणे": "pune",
    "लखनऊ": "lucknow",
    "जयपुर": "jaipur",
    "लुधियाना": "ludhiana",
    "चंडीगढ़": "chandigarh",
    "भोपाल": "bhopal",
    "पटना": "patna",
    "अहमदाबाद": "ahmedabad",
    "हैदराबाद": "hyderabad",
    "बेंगलुरु": "bengaluru",
    "कोलकाता": "kolkata",
    "वाराणसी": "varanasi",
    "बनारस": "varanasi",
    "इंदौर": "indore",
    "नागपुर": "nagpur",
    "नासिक": "nashik",
    "अमृतसर": "amritsar",
}

WMO_WEATHER_CODES = {
    0: ("Clear Sky", "☀️"),
    1: ("Mainly Clear", "🌤️"),
    2: ("Partly Cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Foggy", "🌫️"),
    48: ("Depositing Rime Fog", "🌫️"),
    51: ("Light Drizzle", "🌦️"),
    53: ("Moderate Drizzle", "🌦️"),
    55: ("Dense Drizzle", "🌧️"),
    61: ("Slight Rain", "🌧️"),
    63: ("Moderate Rain", "🌧️"),
    65: ("Heavy Rain", "🌧️"),
    71: ("Slight Snow", "❄️"),
    73: ("Moderate Snow", "❄️"),
    75: ("Heavy Snow", "❄️"),
    80: ("Slight Rain Showers", "🌦️"),
    81: ("Moderate Rain Showers", "🌧️"),
    82: ("Violent Rain Showers", "⛈️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with Slight Hail", "⛈️"),
    99: ("Thunderstorm with Heavy Hail", "⛈️"),
}


def _interpret_wmo(code: int) -> Tuple[str, str]:
    """Return condition text and icon for WMO weather code."""
    return WMO_WEATHER_CODES.get(code, ("Partly Cloudy", "⛅"))


def _generate_agronomic_advisory(
    max_t: float,
    min_t: float,
    rain_mm: float,
    rain_prob: int,
    wind_kmh: float,
    condition: str,
) -> str:
    """Generate dynamic actionable farming advisory based on real weather metrics."""
    if rain_mm >= 15 or rain_prob >= 75:
        return (
            f"Heavy rainfall expected ({rain_mm:.1f} mm). Ensure rapid field drainage "
            "to prevent waterlogging. Halt all irrigation, sowing, and foliar spray operations."
        )
    if rain_mm >= 3 or rain_prob >= 45:
        return (
            f"Rain forecast ({rain_mm:.1f} mm, {rain_prob}% chance). Hold off scheduled irrigation. "
            "Delay fertilizer broadcasting and chemical spraying until foliage dries."
        )
    if wind_kmh >= 20:
        return (
            f"Gusty winds ({wind_kmh:.1f} km/h). Strictly avoid pesticide and herbicide spraying "
            "to prevent drift damage. Provide mechanical staking to fragile crops like banana or tomato."
        )
    if max_t >= 38:
        return (
            f"High heat stress ({max_t:.1f}°C). Irrigate during late evening or early morning to reduce "
            "evaporation. Apply mulch to conserve soil moisture."
        )
    if min_t <= 6:
        return (
            f"Cold wave warning (Min {min_t:.1f}°C). High frost risk for mustard, potato, and vegetables. "
            "Provide light night irrigation or smoke cover to protect young canopies."
        )
    if min_t <= 12 and rain_mm == 0:
        return (
            f"Cool, dry weather ({min_t:.1f}–{max_t:.1f}°C). Excellent conditions for rabi wheat growth "
            "and timely weeding. Morning hours suitable for micronutrient foliar application."
        )
    return (
        f"Favorable conditions ({min_t:.1f}–{max_t:.1f}°C, Wind {wind_kmh:.1f} km/h). "
        "Ideal window for field inspection, routine weeding, and balanced fertilizer top-dressing."
    )


class WeatherService:
    """Provides real-time live weather and 7-day agricultural forecasts via Open-Meteo."""

    def __init__(self, cache_ttl_seconds: int = 900):
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._cache_ttl = cache_ttl_seconds

    def geocode_city(self, location_name: str) -> Optional[Tuple[float, float, str, str]]:
        """
        Resolve city name to (lat, lon, standard_city_name, state).
        Uses local dictionary first, then Open-Meteo Geocoding API.
        """
        norm = location_name.strip().lower()

        # Check Hindi mapping
        for h_name, std_name in CITY_TRANSLATIONS.items():
            if h_name in norm:
                norm = std_name
                break

        # Check predefined major Indian cities
        for city_key, (lat, lon, state) in MAJOR_CITIES.items():
            if city_key in norm or norm in city_key:
                return (lat, lon, city_key.title(), state)

        # Fallback to Open-Meteo Geocoding API
        try:
            url = (
                "https://geocoding-api.open-meteo.com/v1/search?"
                + urllib.parse.urlencode({
                    "name": location_name.strip(),
                    "count": 1,
                    "language": "en",
                    "format": "json",
                })
            )
            req = urllib.request.Request(url, headers={"User-Agent": "KrishiRakshak/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("results")
                if results and len(results) > 0:
                    r = results[0]
                    lat = float(r["latitude"])
                    lon = float(r["longitude"])
                    c_name = r.get("name", location_name.title())
                    state = r.get("admin1", r.get("country", "India"))
                    return (lat, lon, c_name, state)
        except Exception as e:
            logger.warning(f"Geocoding failed for '{location_name}': {e}")

        # Ultimate fallback to Delhi
        return (28.6139, 77.2090, "Delhi", "Delhi")

    def get_live_weather(self, location: str = "Delhi") -> Dict[str, Any]:
        """
        Fetch real-time weather and 7-day forecast for the given location.
        Returns cached data if within TTL.
        """
        geo = self.geocode_city(location)
        if not geo:
            geo = (28.6139, 77.2090, "Delhi", "Delhi")

        lat, lon, city_name, state = geo
        cache_key = f"{lat:.3f}_{lon:.3f}"
        now = time.time()

        if cache_key in self._cache:
            ts, cached_data = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                return cached_data

        try:
            params = {
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,weather_code,wind_speed_10m"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
                ),
                "timezone": "Asia/Kolkata",
                "forecast_days": 7,
            }
            url = f"https://api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": "KrishiRakshak/1.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                raw = json.loads(resp.read().decode())

            curr = raw.get("current", {})
            daily = raw.get("daily", {})

            curr_code = int(curr.get("weather_code", 0))
            curr_cond, curr_icon = _interpret_wmo(curr_code)

            days_list = []
            dates = daily.get("time", [])
            w_codes = daily.get("weather_code", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            precips = daily.get("precipitation_sum", [])
            precip_probs = daily.get("precipitation_probability_max", [])
            wind_maxs = daily.get("wind_speed_10m_max", [])

            for i in range(len(dates)):
                d_str = dates[i]
                try:
                    dt = datetime.strptime(d_str, "%Y-%m-%d")
                    day_name = dt.strftime("%A")
                except Exception:
                    day_name = f"Day {i+1}"

                code = int(w_codes[i]) if i < len(w_codes) else 0
                cond, icon = _interpret_wmo(code)
                max_t = float(max_temps[i]) if i < len(max_temps) else 25.0
                min_t = float(min_temps[i]) if i < len(min_temps) else 15.0
                rain_mm = float(precips[i]) if i < len(precips) else 0.0
                rain_prob = int(precip_probs[i]) if (precip_probs and i < len(precip_probs) and precip_probs[i] is not None) else 0
                wind_kmh = float(wind_maxs[i]) if i < len(wind_maxs) else 10.0

                advisory = _generate_agronomic_advisory(
                    max_t=max_t,
                    min_t=min_t,
                    rain_mm=rain_mm,
                    rain_prob=rain_prob,
                    wind_kmh=wind_kmh,
                    condition=cond,
                )

                days_list.append({
                    "date": d_str,
                    "day": day_name,
                    "temperature": {
                        "min": round(min_t, 1),
                        "max": round(max_t, 1),
                        "unit": "Celsius",
                    },
                    "humidity": int(curr.get("relative_humidity_2m", 60)) if i == 0 else 65,
                    "rainfall": round(rain_mm, 1),
                    "rain_probability": rain_prob,
                    "wind_speed": round(wind_kmh, 1),
                    "condition": cond,
                    "icon": icon,
                    "advisory": advisory,
                })

            today_advisory = days_list[0]["advisory"] if days_list else "Favorable farming conditions."

            # Seasonal crop impact analysis
            curr_temp = float(curr.get("temperature_2m", 22.0))
            if curr_temp < 15:
                crop_impact = "Cool winter climate active. Favorable for rabi cereals (wheat, barley, mustard). Protect sensitive leafy crops from night chilling."
            elif curr_temp > 35:
                crop_impact = "Intense summer heat. Provide frequent light irrigations and monitor for moisture stress in summer pulses and vegetables."
            else:
                crop_impact = "Moderate ambient conditions. Active growth phase for crops. Normal irrigation and nutrient schedules recommended."

            live_result = {
                "is_live": True,
                "source": "Open-Meteo Real-Time Agricultural API",
                "city": city_name,
                "state": state,
                "latitude": lat,
                "longitude": lon,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
                "current": {
                    "temperature": round(curr_temp, 1),
                    "apparent_temperature": round(float(curr.get("apparent_temperature", curr_temp)), 1),
                    "humidity": int(curr.get("relative_humidity_2m", 60)),
                    "precipitation": round(float(curr.get("precipitation", 0.0)), 1),
                    "wind_speed": round(float(curr.get("wind_speed_10m", 5.0)), 1),
                    "condition": curr_cond,
                    "icon": curr_icon,
                    "advisory": today_advisory,
                },
                "forecasts": days_list,
                "crop_impact": crop_impact,
            }

            self._cache[cache_key] = (now, live_result)
            return live_result

        except Exception as e:
            logger.error(f"Failed to fetch live weather from Open-Meteo: {e}")
            # Fallback to local baseline data if network fails
            return self._fallback_weather(city_name, state)

    def _fallback_weather(self, city: str, state: str) -> Dict[str, Any]:
        """Graceful fallback if internet is completely disconnected."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        return {
            "is_live": False,
            "source": "Offline Baseline Data",
            "city": city,
            "state": state,
            "last_updated": f"{today_str} (Offline)",
            "current": {
                "temperature": 22.0,
                "apparent_temperature": 23.0,
                "humidity": 65,
                "precipitation": 0.0,
                "wind_speed": 6.0,
                "condition": "Partly Cloudy",
                "icon": "⛅",
                "advisory": "Weather data offline. General advice: Maintain regular field monitoring and check soil moisture.",
            },
            "forecasts": [
                {
                    "date": today_str,
                    "day": "Today",
                    "temperature": {"min": 16.0, "max": 28.0, "unit": "Celsius"},
                    "humidity": 65,
                    "rainfall": 0.0,
                    "rain_probability": 10,
                    "wind_speed": 7.0,
                    "condition": "Partly Cloudy",
                    "icon": "⛅",
                    "advisory": "Maintain regular field monitoring and check soil moisture.",
                }
            ],
            "crop_impact": "Monitor local conditions and consult nearest Krishi Vigyan Kendra (KVK).",
        }


# Global singleton instance
_weather_service_instance: Optional[WeatherService] = None


def get_weather_service() -> WeatherService:
    """Retrieve singleton WeatherService instance."""
    global _weather_service_instance
    if _weather_service_instance is None:
        _weather_service_instance = WeatherService()
    return _weather_service_instance

