"""
Climate Risk Predictor & Early Warning System Service
Leverages real-time meteorological observations, historical 7-day weather,
satellite-derived root zone soil moisture (0-1cm, 1-3cm, 3-9cm, 9-27cm),
and crop growth phenology to forecast agronomic climate threats:
- Drought & Soil Moisture Deficit
- Flood & Root Waterlogging
- Terminal Heat Stress
- Frost & Radiative Chilling
- Heavy Rain & Crop Lodging
- Climate-driven Fungal / Pathogen Outbreaks
"""
import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from services.weather_service import get_weather_service

logger = logging.getLogger(__name__)

# Crop Growth Stages and Temperature Sensitivities
CROP_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "wheat": {
        "name": "Wheat (गेहूं / ਕਣਕ)",
        "rabi_crop": True,
        "stages": ["Sowing/CRI", "Vegetative/Tillering", "Booting/Flowering", "Grain Filling/Milking", "Maturity/Harvest"],
        "heat_stress_thresholds": {
            "Sowing/CRI": 32.0,
            "Vegetative/Tillering": 30.0,
            "Booting/Flowering": 28.0,
            "Grain Filling/Milking": 30.0,
            "Maturity/Harvest": 36.0,
        },
        "frost_stress_threshold": 4.0,
        "flood_sensitive_stages": ["Sowing/CRI", "Booting/Flowering"],
        "drought_wilting_moisture": 0.17,
        "ideal_soil_moisture": (0.24, 0.36),
    },
    "rice": {
        "name": "Rice / Paddy (धान / ਝੋਨਾ)",
        "rabi_crop": False,
        "stages": ["Nursery/Transplanting", "Tillering", "Panicle Initiation/Flowering", "Milking/Dough", "Maturity/Harvest"],
        "heat_stress_thresholds": {
            "Nursery/Transplanting": 38.0,
            "Tillering": 36.0,
            "Panicle Initiation/Flowering": 35.0,
            "Milking/Dough": 36.0,
            "Maturity/Harvest": 40.0,
        },
        "frost_stress_threshold": 10.0,
        "flood_sensitive_stages": ["Nursery/Transplanting", "Maturity/Harvest"],
        "drought_wilting_moisture": 0.22,
        "ideal_soil_moisture": (0.32, 0.45),
    },
    "tomato": {
        "name": "Tomato (टमाटर / ਟਮਾਟਰ)",
        "rabi_crop": True,
        "stages": ["Nursery/Transplanting", "Vegetative", "Flowering", "Fruit Development", "Harvesting"],
        "heat_stress_thresholds": {
            "Nursery/Transplanting": 33.0,
            "Vegetative": 34.0,
            "Flowering": 32.0,
            "Fruit Development": 34.0,
            "Harvesting": 36.0,
        },
        "frost_stress_threshold": 6.0,
        "flood_sensitive_stages": ["Nursery/Transplanting", "Flowering", "Fruit Development"],
        "drought_wilting_moisture": 0.18,
        "ideal_soil_moisture": (0.22, 0.35),
    },
    "potato": {
        "name": "Potato (आलू / ਆਲੂ)",
        "rabi_crop": True,
        "stages": ["Sprouting/Emergence", "Vegetative Growth", "Tuber Initiation", "Tuber Bulking", "Senescence/Harvest"],
        "heat_stress_thresholds": {
            "Sprouting/Emergence": 28.0,
            "Vegetative Growth": 26.0,
            "Tuber Initiation": 24.0,
            "Tuber Bulking": 26.0,
            "Senescence/Harvest": 30.0,
        },
        "frost_stress_threshold": 4.0,
        "flood_sensitive_stages": ["Tuber Initiation", "Tuber Bulking"],
        "drought_wilting_moisture": 0.19,
        "ideal_soil_moisture": (0.25, 0.38),
    },
    "mustard": {
        "name": "Mustard (सरसों / ਸਰ੍ਹੋਂ)",
        "rabi_crop": True,
        "stages": ["Germination/Seedling", "Vegetative Growth", "Flowering", "Siliqua/Pod Filling", "Maturity"],
        "heat_stress_thresholds": {
            "Germination/Seedling": 32.0,
            "Vegetative Growth": 28.0,
            "Flowering": 26.0,
            "Siliqua/Pod Filling": 28.0,
            "Maturity": 34.0,
        },
        "frost_stress_threshold": 3.5,
        "flood_sensitive_stages": ["Germination/Seedling", "Flowering"],
        "drought_wilting_moisture": 0.15,
        "ideal_soil_moisture": (0.20, 0.32),
    },
    "maize": {
        "name": "Maize (मक्का / ਮੱਕੀ)",
        "rabi_crop": False,
        "stages": ["Germination/Seedling", "Knee High/Vegetative", "Tasseling/Silking", "Grain Filling", "Maturity"],
        "heat_stress_thresholds": {
            "Germination/Seedling": 36.0,
            "Knee High/Vegetative": 38.0,
            "Tasseling/Silking": 35.0,
            "Grain Filling": 36.0,
            "Maturity": 40.0,
        },
        "frost_stress_threshold": 8.0,
        "flood_sensitive_stages": ["Germination/Seedling", "Tasseling/Silking"],
        "drought_wilting_moisture": 0.18,
        "ideal_soil_moisture": (0.22, 0.35),
    },
    "cotton": {
        "name": "Cotton (कपास / ਨਰਮਾ)",
        "rabi_crop": False,
        "stages": ["Seedling", "Vegetative/Square Formation", "Flowering/Boll Formation", "Boll Opening", "Harvesting"],
        "heat_stress_thresholds": {
            "Seedling": 40.0,
            "Vegetative/Square Formation": 38.0,
            "Flowering/Boll Formation": 37.0,
            "Boll Opening": 40.0,
            "Harvesting": 42.0,
        },
        "frost_stress_threshold": 12.0,
        "flood_sensitive_stages": ["Seedling", "Flowering/Boll Formation"],
        "drought_wilting_moisture": 0.16,
        "ideal_soil_moisture": (0.22, 0.35),
    },
}


class ClimateRiskService:
    """Calculates multidimensional climate risk scores and early warning alerts."""

    def __init__(self, cache_ttl_seconds: int = 600):
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._cache_ttl = cache_ttl_seconds

    def _fetch_satellite_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Query Open-Meteo for past 7 days historical weather, current conditions,
        satellite-derived 4-layer soil moisture, and next 7 days forecast.
        """
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m,soil_temperature_0cm,"
                "soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,"
                "soil_moisture_3_to_9cm,soil_moisture_9_to_27cm"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
            ),
            "past_days": 7,
            "forecast_days": 7,
            "timezone": "Asia/Kolkata",
        }
        url = f"https://api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "KrishiRakshak/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode())

    def assess_crop_climate_risk(
        self,
        location: str = "Delhi",
        crop: str = "wheat",
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform multidimensional climate risk prediction and generate early warning alerts.
        """
        norm_crop = crop.strip().lower()
        if "wheat" in norm_crop or "gehu" in norm_crop or "गेहूं" in norm_crop or "ਕਣਕ" in norm_crop:
            crop_key = "wheat"
        elif "rice" in norm_crop or "paddy" in norm_crop or "धान" in norm_crop or "चावल" in norm_crop or "ਝੋਨਾ" in norm_crop:
            crop_key = "rice"
        elif "tomato" in norm_crop or "tamatar" in norm_crop or "टमाटर" in norm_crop:
            crop_key = "tomato"
        elif "potato" in norm_crop or "aloo" in norm_crop or "आलू" in norm_crop:
            crop_key = "potato"
        elif "mustard" in norm_crop or "sarson" in norm_crop or "सरसों" in norm_crop:
            crop_key = "mustard"
        elif "maize" in norm_crop or "makka" in norm_crop or "मक्का" in norm_crop:
            crop_key = "maize"
        elif "cotton" in norm_crop or "kapas" in norm_crop or "कपास" in norm_crop:
            crop_key = "cotton"
        else:
            crop_key = "wheat"

        crop_info = CROP_THRESHOLDS[crop_key]
        stages_list = crop_info["stages"]

        if not stage or stage not in stages_list:
            stage = stages_list[2]  # Default to sensitive flowering/reproductive stage

        # Geocode location
        ws = get_weather_service()
        geo = ws.geocode_city(location) or (28.6139, 77.2090, "Delhi", "Delhi")
        lat, lon, city_name, state_name = geo

        cache_key = f"{lat:.2f}_{lon:.2f}_{crop_key}_{stage}"
        now = time.time()
        if cache_key in self._cache:
            ts, val = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                return val

        # Fetch real-time + historical weather & satellite soil moisture
        try:
            raw_data = self._fetch_satellite_weather_data(lat, lon)
        except Exception as e:
            logger.error(f"Climate risk data fetch failed: {e}")
            return self._build_offline_fallback(city_name, state_name, crop_info, stage)

        curr = raw_data.get("current", {})
        daily = raw_data.get("daily", {})

        # Soil moisture layers (in m3/m3)
        sm_surface = float(curr.get("soil_moisture_0_to_1cm", 0.28))
        sm_shallow = float(curr.get("soil_moisture_1_to_3cm", 0.29))
        sm_root_act = float(curr.get("soil_moisture_3_to_9cm", 0.31))
        sm_deep = float(curr.get("soil_moisture_9_to_27cm", 0.33))
        avg_root_moisture = (sm_shallow + sm_root_act * 2 + sm_deep) / 4.0

        daily_precip = daily.get("precipitation_sum", [])
        daily_max_t = daily.get("temperature_2m_max", [])
        daily_min_t = daily.get("temperature_2m_min", [])
        daily_wind = daily.get("wind_speed_10m_max", [])
        daily_dates = daily.get("time", [])

        # Partition 14 days into past 7 days and future 7 days
        past_precip = daily_precip[:7] if len(daily_precip) >= 7 else [0.0] * 7
        future_precip = daily_precip[7:] if len(daily_precip) >= 14 else daily_precip
        past_max_t = daily_max_t[:7] if len(daily_max_t) >= 7 else [25.0] * 7
        future_max_t = daily_max_t[7:] if len(daily_max_t) >= 14 else daily_max_t
        future_min_t = daily_min_t[7:] if len(daily_min_t) >= 14 else daily_min_t
        future_wind = daily_wind[7:] if len(daily_wind) >= 14 else daily_wind

        past_rain_sum = sum(past_precip)
        future_rain_sum = sum(future_precip)
        max_forecast_24h_rain = max(future_precip) if future_precip else 0.0
        max_forecast_temp = max(future_max_t) if future_max_t else 28.0
        min_forecast_temp = min(future_min_t) if future_min_t else 15.0
        max_forecast_wind = max(future_wind) if future_wind else 10.0
        curr_humidity = float(curr.get("relative_humidity_2m", 65))

        # ── 1. Drought Risk Calculation ───────────────────────────────────────
        drought_score = 0.0
        wilting_threshold = crop_info["drought_wilting_moisture"]
        if avg_root_moisture < wilting_threshold:
            drought_score += 55.0
        elif avg_root_moisture < wilting_threshold + 0.05:
            drought_score += 35.0
        elif avg_root_moisture < wilting_threshold + 0.10:
            drought_score += 15.0

        # Rainfall deficit factor
        if future_rain_sum < 2.0 and past_rain_sum < 5.0:
            drought_score += 30.0
        elif future_rain_sum < 5.0:
            drought_score += 15.0

        if max_forecast_temp > 33.0:
            drought_score += 15.0
        drought_score = min(round(drought_score), 100)

        # ── 2. Flood & Waterlogging Risk Calculation ──────────────────────────
        flood_score = 0.0
        if max_forecast_24h_rain >= 50.0:
            flood_score += 60.0
        elif max_forecast_24h_rain >= 25.0:
            flood_score += 35.0
        elif max_forecast_24h_rain >= 15.0:
            flood_score += 15.0

        if future_rain_sum >= 70.0:
            flood_score += 25.0

        if sm_surface >= 0.38 or avg_root_moisture >= 0.39:
            flood_score += 20.0  # Saturated soil

        if stage in crop_info.get("flood_sensitive_stages", []):
            flood_score = int(flood_score * 1.15)
        flood_score = min(round(flood_score), 100)

        # ── 3. Heat Stress Calculation ────────────────────────────────────────
        heat_score = 0.0
        stage_heat_thresh = crop_info["heat_stress_thresholds"].get(stage, 32.0)
        temp_delta = max_forecast_temp - stage_heat_thresh

        if temp_delta >= 4.0:
            heat_score = 85.0
        elif temp_delta >= 2.0:
            heat_score = 65.0
        elif temp_delta >= 0.0:
            heat_score = 40.0
        elif temp_delta >= -2.0:
            heat_score = 20.0
        else:
            heat_score = 5.0
        heat_score = min(round(heat_score), 100)

        # ── 4. Frost / Cold Wave Risk Calculation ─────────────────────────────
        frost_score = 0.0
        frost_thresh = crop_info["frost_stress_threshold"]
        if min_forecast_temp <= frost_thresh - 2.0:
            frost_score = 90.0
        elif min_forecast_temp <= frost_thresh:
            frost_score = 65.0
        elif min_forecast_temp <= frost_thresh + 3.0:
            frost_score = 30.0
        else:
            frost_score = 5.0
        frost_score = min(round(frost_score), 100)

        # ── 5. Heavy Rain & Crop Lodging Risk ─────────────────────────────────
        lodging_score = 0.0
        if max_forecast_wind >= 25.0 and max_forecast_24h_rain >= 15.0:
            lodging_score = 80.0
        elif max_forecast_wind >= 20.0 or max_forecast_24h_rain >= 20.0:
            lodging_score = 50.0
        elif max_forecast_wind >= 16.0:
            lodging_score = 25.0
        else:
            lodging_score = 5.0

        if "Maturity" in stage or "Grain Filling" in stage or "Tuber Bulking" in stage:
            lodging_score = int(lodging_score * 1.2)
        lodging_score = min(round(lodging_score), 100)

        # ── 6. Fungal & Pest Climate Risk ─────────────────────────────────────
        fungal_score = 0.0
        if curr_humidity >= 80 and 16.0 <= max_forecast_temp <= 28.0:
            fungal_score = 75.0
        elif curr_humidity >= 70 and 15.0 <= max_forecast_temp <= 30.0:
            fungal_score = 50.0
        elif curr_humidity >= 60:
            fungal_score = 25.0
        else:
            fungal_score = 10.0
        fungal_score = min(round(fungal_score), 100)

        # Calculate Overall Climate Vulnerability Index (CVI)
        all_risks = [drought_score, flood_score, heat_score, frost_score, lodging_score, fungal_score]
        cvi = round(max(all_risks) * 0.65 + (sum(all_risks) / len(all_risks)) * 0.35)

        def _severity(score: int) -> Tuple[str, str]:
            if score >= 70: return ("SEVERE ALERT", "🔴")
            if score >= 45: return ("HIGH RISK", "🟠")
            if score >= 25: return ("MODERATE WATCH", "🟡")
            return ("LOW / NORMAL", "🟢")

        # Compile Early Warning Alerts & Specific Mitigation Actions
        active_alerts: List[Dict[str, Any]] = []
        preventive_actions: List[str] = []

        if drought_score >= 45:
            sev_text, icon = _severity(drought_score)
            active_alerts.append({
                "type": "Drought & Moisture Stress",
                "severity": sev_text,
                "icon": "🏜️",
                "score": drought_score,
                "message": f"Root zone soil moisture is low ({avg_root_moisture:.3f} m³/m³). Forecast rainfall is minimal ({future_rain_sum:.1f} mm).",
            })
            preventive_actions.append(
                f"• **Moisture Conservation**: Schedule light evening drip/sprinkler irrigation for {crop_info['name']}. Apply straw or crop residue mulch to preserve soil moisture."
            )
            preventive_actions.append(
                "• **Anti-Transpirant Foliar Spray**: In case of prolonged dry spell, spray Kaolin clay (5%) or Potassium Chloride (1%) to reduce leaf transpiration."
            )

        if flood_score >= 40:
            sev_text, icon = _severity(flood_score)
            active_alerts.append({
                "type": "Flood & Waterlogging Hazard",
                "severity": sev_text,
                "icon": "🌊",
                "score": flood_score,
                "message": f"Excess rainfall expected (up to {max_forecast_24h_rain:.1f} mm in 24h). Risk of root zone asphyxiation during {stage}.",
            })
            preventive_actions.append(
                f"• **Drainage Clearance**: Dig cross-trenches and clear field furrows to prevent standing water in {crop_info['name']} plots."
            )
            preventive_actions.append(
                "• **Nutrient Management**: Immediately halt nitrogen (urea) top-dressing and chemical sprays until water drains to prevent root scorch and leaching."
            )

        if heat_score >= 45:
            sev_text, icon = _severity(heat_score)
            active_alerts.append({
                "type": "Terminal Heat Stress Warning",
                "severity": sev_text,
                "icon": "🌡️",
                "score": heat_score,
                "message": f"Maximum temperature will touch {max_forecast_temp:.1f}°C, exceeding critical threshold ({stage_heat_thresh}°C) for {stage}.",
            })
            preventive_actions.append(
                f"• **Heat Shock Mitigation**: Foliar spray 1% Potassium Nitrate (KNO₃) or Salicylic Acid (100 ppm) on {crop_info['name']} canopy to induce thermotolerance."
            )
            preventive_actions.append(
                "• **Microclimate Cooling**: Provide light evening irrigation during reproductive flowering to lower the plant canopy temperature by 2–3°C."
            )

        if frost_score >= 45:
            sev_text, icon = _severity(frost_score)
            active_alerts.append({
                "type": "Frost & Cold Wave Alert",
                "severity": sev_text,
                "icon": "❄️",
                "score": frost_score,
                "message": f"Minimum temperature dropping to {min_forecast_temp:.1f}°C. Severe radiative cooling expected at night.",
            })
            preventive_actions.append(
                f"• **Thermal Protection**: Provide light nighttime irrigation to release latent heat and raise canopy temperature by 1.5–2°C."
            )
            preventive_actions.append(
                "• **Night Smudging**: Burn farm residues/smoke heaps along north-western field borders from 2:00 AM to 6:00 AM to trap outgoing radiation."
            )

        if lodging_score >= 45:
            sev_text, icon = _severity(lodging_score)
            active_alerts.append({
                "type": "Heavy Rain & Wind Lodging",
                "severity": sev_text,
                "icon": "🌧️",
                "score": lodging_score,
                "message": f"Gusty winds ({max_forecast_wind:.1f} km/h) combined with rain forecast. Stalk breakage risk during {stage}.",
            })
            preventive_actions.append(
                "• **Staking & Earthing-up**: Provide mechanical support to tall crops or fruit trees. Postpone flood irrigation before windy days to prevent wet-soil stem toppling."
            )

        if fungal_score >= 50:
            sev_text, icon = _severity(fungal_score)
            active_alerts.append({
                "type": "Pathogen & Fungal Outbreak Risk",
                "severity": sev_text,
                "icon": "🦠",
                "score": fungal_score,
                "message": f"High humidity ({curr_humidity}%) combined with mild temperatures ({curr.get('temperature_2m')}°C) favors rust and blight multiplication.",
            })
            preventive_actions.append(
                "• **Prophylactic Biocontrol**: Spray Trichoderma viride (@ 2.5 kg/ha) or Mancozeb 75 WP (@ 2g/litre water) as protective barrier on lower foliage."
            )

        if not preventive_actions:
            preventive_actions.append(
                f"• **Optimal Conditions**: Weather parameters are within favorable agronomic ranges for {crop_info['name']} at {stage}. Continue scheduled intercultural operations and balanced nutrition."
            )

        # 14-Day Timeline Data for Visualization
        timeline_data = []
        for i, date_str in enumerate(daily_dates):
            is_past = (i < 7)
            timeline_data.append({
                "date": date_str,
                "type": "Historical (Past)" if is_past else "Forecast (Next 7d)",
                "max_temp": float(daily_max_t[i]) if i < len(daily_max_t) else 25.0,
                "min_temp": float(daily_min_t[i]) if i < len(daily_min_t) else 15.0,
                "rainfall": float(daily_precip[i]) if i < len(daily_precip) else 0.0,
            })

        result = {
            "success": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "location": city_name,
            "state": state_name,
            "latitude": lat,
            "longitude": lon,
            "crop": crop_info["name"],
            "crop_key": crop_key,
            "growth_stage": stage,
            "all_stages": stages_list,
            "climate_vulnerability_index": cvi,
            "cvi_status": _severity(cvi)[0],
            "cvi_icon": _severity(cvi)[1],
            "current_weather": {
                "temperature": float(curr.get("temperature_2m", 22.0)),
                "humidity": curr_humidity,
                "wind_speed": float(curr.get("wind_speed_10m", 5.0)),
                "precipitation": float(curr.get("precipitation", 0.0)),
            },
            "satellite_soil_moisture": {
                "surface_0_to_1cm": sm_surface,
                "seed_zone_1_to_3cm": sm_shallow,
                "active_root_3_to_9cm": sm_root_act,
                "deep_root_9_to_27cm": sm_deep,
                "average_root_moisture": round(avg_root_moisture, 3),
                "unit": "m³/m³ (volumetric)",
                "wilting_threshold": wilting_threshold,
                "status": "Adequate" if avg_root_moisture >= wilting_threshold + 0.06 else ("Marginal" if avg_root_moisture >= wilting_threshold else "Moisture Stressed"),
            },
            "risk_scores": {
                "drought_risk": {"score": drought_score, "level": _severity(drought_score)[0], "icon": "🏜️"},
                "flood_risk": {"score": flood_score, "level": _severity(flood_score)[0], "icon": "🌊"},
                "heat_stress": {"score": heat_score, "level": _severity(heat_score)[0], "icon": "🌡️"},
                "frost_warning": {"score": frost_score, "level": _severity(frost_score)[0], "icon": "❄️"},
                "lodging_risk": {"score": lodging_score, "level": _severity(lodging_score)[0], "icon": "🌧️"},
                "fungal_disease_risk": {"score": fungal_score, "level": _severity(fungal_score)[0], "icon": "🦠"},
            },
            "active_alerts": active_alerts,
            "preventive_actions": preventive_actions,
            "timeline": timeline_data,
        }

        self._cache[cache_key] = (now, result)
        return result

    def _build_offline_fallback(self, city: str, state: str, crop_info: dict, stage: str) -> Dict[str, Any]:
        """Graceful offline fallback if satellite endpoint is temporarily unreachable."""
        return {
            "success": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST (Offline)"),
            "location": city,
            "state": state,
            "crop": crop_info["name"],
            "crop_key": "wheat",
            "growth_stage": stage,
            "all_stages": crop_info["stages"],
            "climate_vulnerability_index": 22,
            "cvi_status": "LOW / NORMAL",
            "cvi_icon": "🟢",
            "current_weather": {"temperature": 22.0, "humidity": 60, "wind_speed": 6.0, "precipitation": 0.0},
            "satellite_soil_moisture": {
                "surface_0_to_1cm": 0.28,
                "seed_zone_1_to_3cm": 0.29,
                "active_root_3_to_9cm": 0.31,
                "deep_root_9_to_27cm": 0.33,
                "average_root_moisture": 0.30,
                "unit": "m³/m³ (volumetric)",
                "wilting_threshold": 0.17,
                "status": "Adequate",
            },
            "risk_scores": {
                "drought_risk": {"score": 15, "level": "LOW / NORMAL", "icon": "🏜️"},
                "flood_risk": {"score": 10, "level": "LOW / NORMAL", "icon": "🌊"},
                "heat_stress": {"score": 20, "level": "LOW / NORMAL", "icon": "🌡️"},
                "frost_warning": {"score": 10, "level": "LOW / NORMAL", "icon": "❄️"},
                "lodging_risk": {"score": 10, "level": "LOW / NORMAL", "icon": "🌧️"},
                "fungal_disease_risk": {"score": 25, "level": "LOW / NORMAL", "icon": "🦠"},
            },
            "active_alerts": [],
            "preventive_actions": [
                f"• Regular monitoring recommended for {crop_info['name']} at {stage}.",
                "• Check soil moisture before initiating next irrigation cycle.",
            ],
            "timeline": [],
        }


# Global singleton
_climate_risk_service: Optional[ClimateRiskService] = None


def get_climate_risk_service() -> ClimateRiskService:
    global _climate_risk_service
    if _climate_risk_service is None:
        _climate_risk_service = ClimateRiskService()
    return _climate_risk_service

