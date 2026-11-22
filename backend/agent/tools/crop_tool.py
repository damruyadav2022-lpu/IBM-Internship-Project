"""
Crop Tool — queries crop_calendar.json to recommend crops by season, crop name, and soil type.
Provides clean, structured, and farmer-focused recommendations without cluttered dumps.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

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


CROP_ALIASES = {
    "wheat": ["wheat", "gehun", "gehu", "गेहूं", "गेहुं", "ਕਣਕ", "গম", "గోధుమ", "गहू", "ghau"],
    "mustard": ["mustard", "sarson", "सरसों", "ਸਰ੍ਹੋਂ", "সরিষা", "ఆవాలు", "मोहरी", "rai"],
    "potato": ["potato", "aloo", "alu", "आलू", "ਆਲੂ", "আলু", "బంగాళాదుంప", "बटाटा", "batata"],
    "rice": ["rice", "paddy", "dhan", "chawal", "धान", "चावल", "ਝੋਨਾ", "ধান", "వరి", "भात", "dangar"],
    "cotton": ["cotton", "kapas", "narma", "कपास", "ਕਪਾਹ", "তুলা", "ప్రత్తి", "कापूस"],
    "maize": ["maize", "corn", "makka", "makki", "मक्का", "ਮੱਕੀ", "ভুট্টা", "మొక్కజొన్న", "मका"],
    "chickpea": ["chickpea", "gram", "chana", "chane", "चना", "ਛੋਲੇ", "ছোলা", "శనగలు", "हरभरा"],
    "pea": ["pea", "peas", "matar", "मटर", "ਮਟਰ", "মটর", "బఠానీ", "मटार"],
    "cucumber": ["cucumber", "kheera", "खीरा", "ਖੀਰਾ", "শসা", "దోసకాయ", "काकडी"],
    "watermelon": ["watermelon", "tarbooz", "तरबूज", "ਤਰਬੂਜ਼", "তরমুজ", "పుచ్చకాయ", "कलिंगड"],
    "soybean": ["soybean", "soya", "soyabean", "सोयाबीन", "ਸੋਇਆਬੀਨ", "সোয়াবিন", "సోయాబీన్"],
    "groundnut": ["groundnut", "peanut", "moongphali", "मूंगफली", "ਮੂੰਗਫਲੀ", "বাদাম", "వేరుశనగ"],
}


def _detect_specific_crop(query: str) -> Optional[str]:
    """Detect if a specific crop is being asked about."""
    q = query.lower()
    for canonical_crop, aliases in CROP_ALIASES.items():
        if any(alias in q for alias in aliases):
            return canonical_crop
    return None


def _detect_season(query: str) -> Optional[str]:
    """Detect season from query text, supporting Hindi names."""
    q = query.lower()
    if any(w in q for w in ["kharif", "monsoon", "rainy", "june", "july", "august", "खरीफ", "मानसून", "ਖਰੀਫ"]):
        return "Kharif"
    if any(w in q for w in ["rabi", "winter", "october", "november", "रबी", "सर्दी", "ठंड", "ਰਬੀ"]):
        return "Rabi"
    if any(w in q for w in ["zaid", "summer", "march", "april", "may", "जायद", "गर्मी", "ਜ਼ਾਇਦ"]):
        return "Zaid"
    return None


def _format_single_crop(crop: Dict[str, Any], season_name: str) -> str:
    """Format a single crop's agronomic profile cleanly and concisely."""
    name_eng = crop.get("name", "")
    name_hi = crop.get("hindi_name", "")
    varieties = ", ".join(crop.get("key_varieties", [])[:3])
    soil = ", ".join(crop.get("soil_types", []))
    water = crop.get("water_requirement", "Medium")
    yield_val = crop.get("expected_yield_per_acre", "N/A")
    duration = crop.get("growing_period_days", "N/A")
    tip = crop.get("planting_tips", "")

    return (
        f"🌱 **{name_eng} ({name_hi}) — कृषि मार्गदर्शिका ({season_name} सीजन)**\n\n"
        f"• 🌾 **उपयुक्त मिट्टी**: {soil}\n"
        f"• 💧 **जल आवश्यकता**: {water} ({duration} दिन की फसल)\n"
        f"• 🌾 **प्रमुख उन्नत किस्में**: {varieties}\n"
        f"• 📈 **अनुमानित उपज**: {yield_val} प्रति एकड़\n"
        f"• 💡 **व्यावहारिक सलाह**: {tip}"
    )


def get_crop_recommendation(query: str) -> str:
    """
    Return clean, structured crop recommendations.
    If a specific crop is mentioned, returns ONLY that crop.
    Otherwise, returns a concise seasonal summary.
    """
    try:
        data = _load_crop_data()
        seasons = data.get("seasons", {})

        # 1. Check if user is asking about a specific crop
        specific_crop_key = _detect_specific_crop(query)
        if specific_crop_key:
            for s_name, s_data in seasons.items():
                for c in s_data.get("crops", []):
                    c_name_lower = c.get("name", "").lower()
                    c_hi_lower = c.get("hindi_name", "").lower()
                    aliases = CROP_ALIASES.get(specific_crop_key, [specific_crop_key])
                    if any(a in c_name_lower or a in c_hi_lower for a in aliases):
                        return _format_single_crop(c, s_name)

        # 2. Season-based recommendations
        season = _detect_season(query)
        if season:
            season_data = seasons.get(season, {})
            crops = season_data.get("crops", [])
            desc = season_data.get("description", "")
            sow = ", ".join(season_data.get("sowing_months", []))
            harv = ", ".join(season_data.get("harvesting_months", []))

            lines = [
                f"🌱 **{season} सीजन — प्रमुख फसलें एवं बुवाई कैलेंडर**",
                f"• 🗓 **बुवाई का समय**: {sow} | **कटाई**: {harv}",
                f"• 📋 **सीजन विवरण**: {desc}",
                "",
                "**सिफारिश की गई प्रमुख फसलें:**",
            ]
            for c in crops[:4]:
                c_name = c.get("name", "")
                c_hi = c.get("hindi_name", "")
                c_yield = c.get("expected_yield_per_acre", "N/A")
                c_vars = ", ".join(c.get("key_varieties", [])[:2])
                lines.append(f"• **{c_name} ({c_hi})**: उपज {c_yield}/एकड़ | किस्में: {c_vars}")

            lines.append("\n💡 *सुझाव: किसी विशिष्ट फसल (जैसे गेहूं, सरसों, आलू) की विस्तृत सलाह के लिए उसका नाम लिखकर पूछें।*")
            return "\n".join(lines)

        # 3. General overview if no specific crop or season detected
        lines = [
            "📅 **भारतीय कृषि सीजन कैलेंडर (त्वरित सारांश)**:",
            "• 🌱 **खरीफ (Kharif)** (जून–जुलाई बुवाई): धान (चावल), मक्का, कपास, सोयाबीन, मूंगफली",
            "• 🌾 **रबी (Rabi)** (अक्टूबर–नवंबर बुवाई): गेहूं, सरसों, चना, आलू, मटर",
            "• 🍉 **जायद (Zaid)** (मार्च–अप्रैल बुवाई): खीरा, ककड़ी, तरबूज, करेला, मूंग",
            "\n💡 *सुझाव: अपनी पसंदीदा फसल या मौसम का नाम लिखें (उदा. 'गेहूं की सलाह' या 'रबी फसलें')।*"
        ]
        return "\n".join(lines)

    except FileNotFoundError:
        return "Crop calendar data currently updating. Please refer to regional agricultural advisory."
    except Exception as e:
        logger.error(f"Crop tool error: {e}")
        return f"Error retrieving crop information: {str(e)}"


def get_crop_tool() -> Tool:
    """Return a LangChain Tool for crop recommendation queries."""
    return Tool(
        name="crop_recommendation",
        description=(
            "Get crop recommendations and agronomic guides by season or specific crop name. "
            "Returns clean varieties, yield, water requirements, and field tips."
        ),
        func=get_crop_recommendation,
    )
