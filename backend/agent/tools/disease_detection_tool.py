"""
Disease Detection Tool for the LangChain farming agent.

This tool allows the agent to interpret disease detection results
from image analysis and provide structured farming advice.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
DISEASE_CLASSES_FILE = DATA_DIR / "disease_classes.json"


def _load_disease_db() -> Dict[str, Any]:
    try:
        with open(DISEASE_CLASSES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.warning(f"Could not load disease classes: {e}")
        return {}


CROP_ALIASES = {
    "टमाटर": "Tomato",
    "ਟਮਾਟਰ": "Tomato",
    "tomato": "Tomato",
    "tamatar": "Tomato",
    "गेहूं": "Wheat",
    "ਕਣਕ": "Wheat",
    "wheat": "Wheat",
    "gehun": "Wheat",
    "धान": "Rice",
    "चावल": "Rice",
    "ਝੋਨਾ": "Rice",
    "rice": "Rice",
    "आलू": "Potato",
    "ਆਲੂ": "Potato",
    "potato": "Potato",
    "aloo": "Potato",
    "मक्का": "Corn",
    "ਮੱਕੀ": "Corn",
    "corn": "Corn",
    "maize": "Corn",
    "सेब": "Apple",
    "ਸੇਬ": "Apple",
    "apple": "Apple",
    "अंगूर": "Grape",
    "ਅੰਗੂਰ": "Grape",
    "grape": "Grape",
    "कसावा": "Cassava",
    "cassava": "Cassava",
}


def query_disease_info(query: str) -> str:
    """
    Query the disease knowledge base for treatment and prevention advice.
    Supports English, Hindi, and Punjabi crop and symptom descriptions.
    """
    db = _load_disease_db()
    diseases = db.get("diseases", {})
    general_advice = db.get("general_advice", {})

    if not diseases:
        return "Disease database not available. Please consult your local agriculture officer."

    query_lower = query.lower().strip()

    # Detect crop from multilingual aliases
    detected_crop = None
    for alias, c_name in CROP_ALIASES.items():
        if alias in query_lower or alias in query:
            detected_crop = c_name
            break

    # ── Find best match ──────────────────────────────────────────────────
    matched_key = None
    matched_disease = None

    # 1. Exact key match
    if query in diseases:
        matched_key = query
        matched_disease = diseases[query]

    # 2. Common name match
    if not matched_disease:
        for key, info in diseases.items():
            if query_lower in info.get("common_name", "").lower():
                matched_key = key
                matched_disease = info
                break

    # 3. If crop detected, prioritize diseases for that crop
    if not matched_disease and detected_crop:
        crop_diseases = []
        for key, info in diseases.items():
            if detected_crop.lower() in info.get("crop", "").lower():
                crop_diseases.append(info)

        if crop_diseases:
            # Check for symptom match in crop diseases (e.g. yellow, blight, rust, spots)
            for d in crop_diseases:
                symptoms_text = (d.get("symptoms", "") + " " + d.get("common_name", "")).lower()
                if any(w in query_lower or w in query for w in ["पीले", "ਪੀਲੇ", "yellow", "धब्बे", "ਧੱਬੇ", "spots"]):
                    if "yellow" in symptoms_text or "blight" in symptoms_text or "spot" in symptoms_text:
                        matched_disease = d
                        break

            if not matched_disease:
                lines = [f"Diseases affecting {detected_crop}:\n"]
                for d in crop_diseases[:4]:
                    severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "none": "🟢"}.get(
                        d.get("severity", "unknown"), "⚪"
                    )
                    lines.append(
                        f"{severity_icon} **{d['common_name']}** ({d.get('hindi_name', '')})\n"
                        f"   Symptoms: {d.get('symptoms', 'N/A')}\n"
                        f"   Organic treatment: {d.get('organic_treatment', 'N/A')}\n"
                        f"   Chemical treatment: {d.get('chemical_treatment', 'N/A')}\n"
                    )
                lines.append(f"\n📞 {general_advice.get('helpline', '')}")
                return "\n".join(lines)

    # 4. Symptom keyword match
    if not matched_disease:
        for key, info in diseases.items():
            symptoms_text = info.get("symptoms", "").lower()
            if any(word in symptoms_text for word in query_lower.split() if len(word) > 3):
                matched_key = key
                matched_disease = info
                break

    if not matched_disease:
        # Return general crop health advice
        return (
            f"No specific disease found matching '{query}'. "
            f"General advice: {general_advice.get('ipm_principle', '')} "
            f"Helpline: {general_advice.get('helpline', '1800-180-1551')}"
        )

    # ── Format detailed response ─────────────────────────────────────────
    d = matched_disease
    severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "none": "🟢"}.get(
        d.get("severity", "unknown"), "⚪"
    )

    response = f"""
{severity_icon} **{d['common_name']}** | {d.get('hindi_name', '')}
Crop: {d.get('crop', 'N/A')} | Severity: {d.get('severity', 'N/A').upper()}

**Symptoms:** {d.get('symptoms', 'N/A')}

**Cause:** {d.get('cause', 'N/A')}

**Favorable Conditions:** {d.get('favorable_conditions', 'N/A')}

**Organic / Biological Treatment:**
{d.get('organic_treatment', 'N/A')}

**Chemical Treatment:**
{d.get('chemical_treatment', 'N/A')}

**Prevention:**
{d.get('prevention', 'N/A')}

**Economic Impact:** {d.get('economic_impact', 'N/A')}

📞 Kisan Helpline: {general_advice.get('helpline', '1800-180-1551')}
⚠️  {general_advice.get('resistance_management', '')}
""".strip()

    return response


def create_disease_tool() -> Tool:
    """Create and return the LangChain Disease Detection Tool."""
    return Tool(
        name="disease_info",
        func=query_disease_info,
        description=(
            "Use this tool to get detailed information about plant diseases, "
            "symptoms, treatments, and prevention. "
            "Input can be: a disease name (e.g. 'Tomato Late Blight'), "
            "a crop name (e.g. 'wheat'), or symptom description "
            "(e.g. 'yellow spots on leaves'). "
            "Also use this when interpreting results from the disease detection image scan."
        ),
    )
