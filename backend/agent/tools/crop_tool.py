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


def _format_single_crop(crop: Dict[str, Any], season_name: str, is_irrigation: bool = False) -> str:
    """Format a single crop's agronomic profile or irrigation schedule cleanly."""
    name_eng = crop.get("name", "")
    name_hi = crop.get("hindi_name", "")
    varieties = ", ".join(crop.get("key_varieties", [])[:3])
    soil = ", ".join(crop.get("soil_types", []))
    water = crop.get("water_requirement", "Medium")
    yield_val = crop.get("expected_yield_per_acre", "N/A")
    duration = crop.get("growing_period_days", "N/A")
    tip = crop.get("planting_tips", "")

    if is_irrigation and "wheat" in name_eng.lower():
        return (
            f"🌾 **{name_eng} ({name_hi}) — संपूर्ण सिंचाई प्रबंधन एवं जल-संसाधन गाइड**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🎯 **१. पहली सिंचाई (CRI अवस्था — शिखर जड़ अवस्था)**:\n"
            f"• **सर्वाधिक संवेदनशील समय**: बुवाई के **20 से 25 दिन बाद** (Crown Root Initiation).\n"
            f"• **वैज्ञानिक महत्व**: इस समय पौधे के तने के आधार से प्राथमिक ताज-जड़ें (crown roots) निकलती हैं। इस समय पानी की कमी होने से कल्ले कम फूटते हैं और उपज में 30% से 40% तक भारी गिरावट आ सकती है।\n"
            f"• **सिंचाई का तरीका**: पहली सिंचाई हमेशा हल्की (3–4 सेमी गहराई) करें ताकि पौधे दबे नहीं।\n\n"
            f"📌 **२. गेहूं की संपूर्ण 5–6 सिंचाई समय-सारणी (Critical Growth Stages)**:\n"
            f"• **१. शिखर जड़ अवस्था (CRI)**: 20–25 दिन बाद (पहला पानी — जीवनरक्षक)\n"
            f"• **२. कल्ले फूटने पर (Tillering)**: 40–45 दिन बाद (दूसरा पानी — कल्ले बढ़ाने हेतु)\n"
            f"• **३. गांठ बनने की अवस्था (Late Jointing)**: 60–65 दिन बाद (तीसरा पानी — तने की मजबूती हेतु)\n"
            f"• **४. फूल आने व बाली निकलते समय (Flowering)**: 80–85 दिन बाद (चौथा पानी — परागण सुरक्षा)\n"
            f"• **५. दूधिया अवस्था (Milking Stage)**: 100–105 दिन बाद (पांचवां पानी — दाने के भराव हेतु)\n"
            f"• **६. दाना सख्त होते समय (Dough Stage)**: 115–120 दिन बाद (यदि गर्म पछुआ हवाएं चलें)\n\n"
            f"🧪 **३. सिंचाई के साथ उर्वरक प्रबंधन (Fertilizer Top-Dressing)**:\n"
            f"• पहली सिंचाई के 2–3 दिन बाद जब खेत में पैर टिकने लगे (ओट आने पर), तब यूरिया की पहली टॉप-ड्रेसिंग (65 किग्रा प्रति एकड़) समान रूप से बिखरें।\n\n"
            f"💡 **४. कृषि वैज्ञानिक सलाह एवं सावधानियां (Pro-Tips)**:\n"
            f"• तेज हवा चलने के समय सिंचाई न करें, अन्यथा फसल गिरने (lodging) का खतरा बढ़ जाता है।\n"
            f"• यदि मिट्टी भारी दोमट है तो 4 सिंचाइयों में भी काम चल सकता है, परंतु रेतीली मिट्टी में 6 सिंचाइयां आवश्यक होती हैं।"
        )

    return (
        f"🌱 **{name_eng} ({name_hi}) — संपूर्ण सस्य प्रबंधन एवं पैकेज ऑफ प्रैक्टिसेज ({season_name} सीजन)**\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🌾 **१. उपयुक्त मिट्टी एवं खेत की तैयारी (Soil & Land Prep)**:\n"
        f"• **मिट्टी**: {soil}। जल निकास की उत्तम व्यवस्था होनी चाहिए।\n"
        f"• **जुताई**: 1 बार मिट्टी पलटने वाले हल से और 2-3 बार कल्टीवेटर से जुताई कर पाटा लगाएं।\n\n"
        f"🗓 **२. बुवाई समय एवं बीज दर (Sowing & Seed Rate)**:\n"
        f"• **फसल अवधि**: {duration} दिन।\n"
        f"• **पंक्ति से पंक्ति दूरी**: 20 से 22.5 सेमी, पौधे से पौधा 10 सेमी, बीज गहराई 4–5 सेमी।\n"
        f"• **बीज शोधन**: कार्बेन्डाजिम या थीरम 2 ग्राम प्रति किग्रा बीज की दर से उपचारित करें।\n\n"
        f"🌾 **३. अनुशंसित उन्नत किस्में (High-Yielding Recommended Varieties)**:\n"
        f"• **प्रमुख किस्में**: {varieties}। (पीली कुंगी/रस्ट प्रतिरोधी एवं अधिक उपज देने वाली)।\n\n"
        f"💧 **४. जल एवं सिंचाई आवश्यकता (Water & Irrigation)**:\n"
        f"• कुल जल मांग: {water}। जीवनरक्षक क्रांतिक अवस्थाओं पर सिंचाई अनिवार्य है।\n\n"
        f"📈 **५. अनुमानित पैदावार (Expected Yield)**:\n"
        f"• **औसत उपज**: {yield_val} प्रति एकड़ (उन्नत प्रबंधन से)।\n\n"
        f"💡 **६. व्यावहारिक किसान सलाह (Agronomist Pro-Tip)**:\n"
        f"• {tip}"
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
        q_lower = query.lower()
        is_irrigation = any(kw in q_lower for kw in ["सिंचाई", "पानी", "irrigation", "water", "पहला पानी", "सींच"])

        # 1. Check if user is asking about a specific crop
        specific_crop_key = _detect_specific_crop(query)
        if specific_crop_key:
            for s_name, s_data in seasons.items():
                for c in s_data.get("crops", []):
                    c_name_lower = c.get("name", "").lower()
                    c_hi_lower = c.get("hindi_name", "").lower()
                    aliases = CROP_ALIASES.get(specific_crop_key, [specific_crop_key])
                    if any(a in c_name_lower or a in c_hi_lower for a in aliases):
                        return _format_single_crop(c, s_name, is_irrigation=is_irrigation)

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
