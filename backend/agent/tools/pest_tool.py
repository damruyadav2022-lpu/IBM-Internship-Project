"""
Pest Tool — queries pest_control.json for pest & disease diagnosis and control remedies.
Delivers direct, actionable, and scientific remedies (organic + chemical with exact dosages).
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "pest_control.json"

_pest_data: Optional[dict] = None


def _load_pest_data() -> dict:
    global _pest_data
    if _pest_data is None:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            _pest_data = json.load(f)
    return _pest_data


CROP_SYNONYMS = {
    "Wheat": ["wheat", "gehun", "gehu", "गेहूं", "गेहुं", "ਕਣਕ", "গম", "గోధుమ", "गहू"],
    "Rice": ["rice", "paddy", "dhan", "chawal", "धान", "चावल", "ਝੋਨਾ", "ধান", "వరి", "भात"],
    "Tomato": ["tomato", "tamatar", "टमाटर", "ਟਮਾਟਰ", "টমেটো", "టమాటా", "टोमॅटो"],
    "Cotton": ["cotton", "kapas", "narma", "कपास", "ਕਪਾਹ", "তুলা", "ప్రత్తి", "कापूस"],
    "Potato": ["potato", "aloo", "alu", "आलू", "ਆਲੂ", "আলু", "बटाटा"],
    "Mustard": ["mustard", "sarson", "सरसों", "ਸਰ੍ਹੋਂ", "সরিষা", "मोहरी"],
}

PEST_KEYWORDS = {
    "Yellow Rust": ["yellow rust", "peela tikka", "कुंगी", "पीली कुंगी", "पीला रतुआ", "रतुआ", "rust", "ਕੁੰਗੀ", "zard kungi"],
    "Aphids": ["aphid", "aphids", "maahu", "teela", "माहू", "चेपा", "तेला", "ਮਾਹੂ", "ਤੇਲਾ", "मोवा"],
    "Termites": ["termite", "termites", "deemak", "दीमक", "ਸਿਉਂਕ", "উইপোকা"],
    "Pink Bollworm": ["pink bollworm", "gulabi sundi", "गुलाबी सुंडी", "सुंडी", "ਗੁਲਾਬੀ ਸੁੰਡੀ", "ਸੁੰਡੀ", "bollworm", "इल्ली"],
    "Whitefly": ["whitefly", "safed makhi", "सफेद मक्खी", "ਚਿੱਟੀ ਮੱਖੀ", "সাদা মাছি"],
    "Tomato Leaf Curl Virus": ["leaf curl", "curl", "patta marodne", "पत्ती मुड़ना", "पत्ता मरोड़", "मरोड़िया", "ਮਰੋੜੀਆ"],
    "Early Blight": ["early blight", "ageti jhulsa", "अगेती झुलसा", "झुलसा", "blight", "ਧੱਬੇ"],
    "Fruit Borer": ["fruit borer", "phal bhedak", "फल छेदक", "ਛੇਦਕ", "borer"],
    "Blast": ["blast", "blast rog", "ब्लास्ट", "ਝੋਨੇ ਦਾ ਬਲਾਸਟ"],
    "Brown Planthopper": ["brown planthopper", "bph", "bhoora tunnabhungi", "भूरा फुदका", "ਭੂਰਾ ਟਿੱਡਾ"],
    "Sheath Blight": ["sheath blight", "sheeth sarakna", "शीथ ब्लाइट", "तने का झुलसा"],
}


def _detect_crop(query: str, pests_by_crop: dict) -> Optional[str]:
    """Detect crop from query using synonyms and keywords."""
    q = query.lower()
    for crop_name, syns in CROP_SYNONYMS.items():
        if any(s in q for s in syns):
            if crop_name in pests_by_crop:
                return crop_name

    # Check direct keys
    for crop in pests_by_crop:
        if crop.lower() in q:
            return crop

    # Default to Wheat if pest matches wheat-specific pests
    if any(k in q for k in ["कुंगी", "रतुआ", "yellow rust"]):
        return "Wheat"
    if any(k in q for k in ["गुलाबी सुंडी", "pink bollworm"]):
        return "Cotton"
    if any(k in q for k in ["पत्ती मुड़ना", "leaf curl", "टमाटर"]):
        return "Tomato"
    if any(k in q for k in ["ब्लास्ट", "blast", "धान"]):
        return "Rice"

    return None


def _format_pest_advice(crop: str, pest: Dict[str, Any]) -> str:
    """Format single pest diagnosis and remedy card cleanly."""
    p_name = pest.get("pest_name", "")
    p_hi = pest.get("hindi_name", "")
    p_type = pest.get("type", "Disease / Pest")
    stage = pest.get("affected_stage", "All Stages")
    symptoms = pest.get("symptoms", "")
    organic = pest.get("organic_remedies", [])
    chemical = pest.get("chemical_remedies", [])
    prevention = pest.get("prevention", [])
    etl = pest.get("economic_threshold", "")

    lines = [
        f"🐛 **{crop} — {p_name} ({p_hi}) नियंत्रण मार्गदर्शिका**",
        "",
        f"🎯 **रोग/कीट पहचान एवं लक्षण ({stage})**:",
        f"• {symptoms}",
    ]

    if organic:
        lines.append("\n🌿 **जैविक व देसी रोकथाम (Organic & IPM)**:")
        for o in organic[:3]:
            lines.append(f"• {o}")

    if chemical:
        lines.append("\n💊 **रासायनिक नियंत्रण एवं सही मात्रा (Chemical Treatment)**:")
        for c in chemical[:2]:
            lines.append(f"• {c}")

    if prevention:
        lines.append("\n🛡️ **सावधानी एवं बचाव उपाय**:")
        for p in prevention[:2]:
            lines.append(f"• {p}")

    if etl and etl != "N/A":
        lines.append(f"\n⚠️ **आर्थिक क्षति स्तर (ETL)**: {etl}")

    lines.append("\n💡 **किसान सलाह**: छिड़काव हमेशा मौसम साफ होने पर और दोपहर बाद शाम के समय करें। सही नोजल का उपयोग करें।")
    return "\n".join(lines)


def get_pest_info(query: str) -> str:
    """
    Return clean, structured pest and disease control information for crops.
    """
    try:
        data = _load_pest_data()
        pests_by_crop = data.get("pests_by_crop", {})
        general = data.get("general_pest_management", {})

        crop = _detect_crop(query, pests_by_crop)
        q = query.lower()

        if not crop:
            # General pest search across all crops
            matched_items = []
            for c_name, p_list in pests_by_crop.items():
                for p in p_list:
                    if any(kw in q for kw in [p["pest_name"].lower(), p.get("hindi_name", "").lower()]):
                        return _format_pest_advice(c_name, p)

            # Return general IPM guide if no specific pest or crop found
            ipm = general.get("IPM_principles", [])
            lines = [
                "🌿 **एकीकृत कीट प्रबंधन (Integrated Pest Management - IPM) प्रमुख सिद्धांत**:",
                "",
            ]
            for tip in ipm[:4]:
                lines.append(f"• {tip}")
            lines.append("\n💡 *सुझाव: विशिष्ट सलाह के लिए फसल और बीमारी/कीट का नाम लिखें (उदा. 'कपास में गुलाबी सुंडी' या 'गेहूं में पीली कुंगी')।*")
            return "\n".join(lines)

        pests = pests_by_crop.get(crop, [])
        if not pests:
            return f"🌾 {crop} फसल के लिए कीट प्रबंधन रिकॉर्ड अद्यतन किए जा रहे हैं।"

        # Match specific pest for this crop
        target_pest = None
        for pest in pests:
            p_name_lower = pest["pest_name"].lower()
            p_hi_lower = pest.get("hindi_name", "").lower()
            if any(term in q for term in [p_name_lower, p_hi_lower]):
                target_pest = pest
                break

        # Also check PEST_KEYWORDS aliases
        if not target_pest:
            for pest_alias, keywords in PEST_KEYWORDS.items():
                if any(kw in q for kw in keywords):
                    for pest in pests:
                        if pest_alias.lower() in pest["pest_name"].lower():
                            target_pest = pest
                            break
                    if target_pest:
                        break

        # Fallback to first pest for this crop if no specific pest matched
        if not target_pest and pests:
            target_pest = pests[0]

        return _format_pest_advice(crop, target_pest)

    except FileNotFoundError:
        return "कीट प्रबंधन डेटा फाइल उपलब्ध नहीं है।"
    except Exception as e:
        logger.error(f"Pest tool error: {e}")
        return f"कीट प्रबंधन जानकारी प्राप्त करने में त्रुटि: {str(e)}"


def get_pest_tool() -> Tool:
    """Return a LangChain Tool for pest and disease queries."""
    return Tool(
        name="pest_disease_control",
        description=(
            "Get pest, insect, and crop disease remedies with organic solutions, "
            "chemical dosages, and field prevention tips. "
            "Use for: rust, bollworm, aphids, blight, termites, leaf curl, etc."
        ),
        func=get_pest_info,
    )
