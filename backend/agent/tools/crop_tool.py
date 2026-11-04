"""
Crop Tool — queries crop_calendar.json to recommend crops by season and soil type.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "crop_calendar.json"

_crop_data: Optional[dict] = None


def _load_crop_data() -> dict:
    global _crop_data
    if _crop_data is None:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            _crop_data = json.load(f)
    return _crop_data


def _detect_season(query: str) -> Optional[str]:
    """Detect season from query text, supporting Hindi names."""
    q = query.lower()
    if any(w in q for w in ["kharif", "monsoon", "rainy", "june", "july", "august", "खरीफ", "मानसून", "बरसात", "धान", "चावल"]):
        return "Kharif"
    if any(w in q for w in ["rabi", "winter", "october", "november", "wheat", "रबी", "सर्दी", "ठंड", "गेहूं", "सरसों"]):
        return "Rabi"
    if any(w in q for w in ["zaid", "summer", "march", "april", "may", "जायद", "गर्मी", "तरबूज", "खीरा"]):
        return "Zaid"
    return None


def _detect_soil(query: str) -> Optional[str]:
    """Detect soil type from query text."""
    q = query.lower()
    soil_keywords = {
        "Black Soil": ["black", "regur", "kali", "cotton soil"],
        "Alluvial Soil": ["alluvial", "domat", "gangetic"],
        "Sandy Soil": ["sandy", "balui", "retili", "sand"],
        "Red Soil": ["red", "laterite", "tambdi"],
        "Loamy Soil": ["loam", "loamy"],
        "Clay Soil": ["clay", "heavy"],
    }
    for soil_type, keywords in soil_keywords.items():
        if any(kw in q for kw in keywords):
            return soil_type
    return None


def get_crop_recommendation(query: str) -> str:
    """
    Return crop recommendations based on season and/or soil type detected in the query.
    """
    try:
        data = _load_crop_data()
        season = _detect_season(query)
        soil = _detect_soil(query)

        results = []

        # Season-based recommendations
        if season:
            season_data = data.get("seasons", {}).get(season, {})
            crops = season_data.get("crops", [])
            desc = season_data.get("description", "")
            results.append(f"🌱 {season} Season Crops")
            results.append(f"📖 {desc}")
            results.append(f"🗓 Sowing: {', '.join(season_data.get('sowing_months', []))}")
            results.append(f"🗓 Harvest: {', '.join(season_data.get('harvesting_months', []))}")
            results.append("")
            for crop in crops[:4]:
                results.append(f"━━━ {crop['name']} ({crop['hindi_name']}) ━━━")
                results.append(f"  Soil: {', '.join(crop['soil_types'])}")
                results.append(f"  Water: {crop['water_requirement']}")
                results.append(f"  Duration: {crop['growing_period_days']} days")
                results.append(f"  Expected Yield: {crop['expected_yield_per_acre']} per acre")
                results.append(f"  Varieties: {', '.join(crop['key_varieties'][:3])}")
                results.append(f"  Tip: {crop['planting_tips']}")
                results.append("")
        else:
            # No season detected — list all seasons briefly
            results.append("📅 Crop Calendar Overview:")
            for s, sdata in data.get("seasons", {}).items():
                crop_names = [c["name"] for c in sdata.get("crops", [])]
                results.append(
                    f"  {s} ({', '.join(sdata.get('sowing_months', []))}): "
                    f"{', '.join(crop_names)}"
                )
            results.append(
                "\nAsk about a specific season (Kharif/Rabi/Zaid) for detailed advice."
            )

        # Soil-based supplement
        if soil:
            soil_mapping = data.get("soil_crop_mapping", {})
            matched = None
            for soil_key, crops in soil_mapping.items():
                if soil.lower() in soil_key.lower():
                    matched = (soil_key, crops)
                    break
            if matched:
                results.append(
                    f"\n🌍 Best crops for {matched[0]}: {', '.join(matched[1])}"
                )

        return "\n".join(results) if results else "No crop data found for your query."

    except FileNotFoundError:
        return "Crop data file not found. Please run the data ingestion script."
    except Exception as e:
        logger.error(f"Crop tool error: {e}")
        return f"Error fetching crop data: {str(e)}"


def get_crop_tool() -> Tool:
    """Return a LangChain Tool for crop recommendation queries."""
    return Tool(
        name="crop_recommendation",
        description=(
            "Get crop recommendations for farmers based on season and soil type. "
            "Provides best crops to grow, planting calendar, expected yield, "
            "suitable varieties, and farming tips. "
            "Use keywords like 'Kharif', 'Rabi', 'Zaid', 'wheat', 'rice', "
            "'black soil', 'sandy soil', etc."
        ),
        func=get_crop_recommendation,
    )
