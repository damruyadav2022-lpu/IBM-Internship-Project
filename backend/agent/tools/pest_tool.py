"""
Pest Tool — queries pest_control.json for pest alerts and remedies.
"""
import json
import logging
from pathlib import Path
from typing import Optional

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


def _find_crop(query: str, pests_by_crop: dict) -> Optional[str]:
    """Case-insensitive crop name detection."""
    q = query.lower()
    for crop in pests_by_crop:
        if crop.lower() in q:
            return crop
    # Common Hindi crop names
    hindi_map = {
        "gehun": "Wheat",
        "gehu": "Wheat",
        "chawal": "Rice",
        "dhan": "Rice",
        "tamatar": "Tomato",
        "kapas": "Cotton",
        "गेहूं": "Wheat",
        "चावल": "Rice",
        "धान": "Rice",
        "टमाटर": "Tomato",
        "कपास": "Cotton",
        "आलू": "Potato",
        "सरसों": "Mustard",
    }
    for hindi, english in hindi_map.items():
        if (hindi in query or hindi in q) and english in pests_by_crop:
            return english
    return None


def get_pest_info(query: str) -> str:
    """
    Return pest control information for a specific crop.
    """
    try:
        data = _load_pest_data()
        pests_by_crop = data.get("pests_by_crop", {})
        general = data.get("general_pest_management", {})

        crop = _find_crop(query, pests_by_crop)

        if not crop:
            crops_available = ", ".join(pests_by_crop.keys())
            ipm = general.get("IPM_principles", [])
            result = [
                f"No specific pest data found for your query.",
                f"Available crops: {crops_available}",
                "",
                "🌿 General IPM Principles:",
            ]
            for tip in ipm[:5]:
                result.append(f"  • {tip}")
            return "\n".join(result)

        pests = pests_by_crop[crop]
        lines = [f"🐛 Pest & Disease Guide for {crop}", "=" * 40, ""]

        # Check if user is asking about a specific pest
        q = query.lower()
        matched_pests = []
        for pest in pests:
            if any(
                kw in q
                for kw in [
                    pest["pest_name"].lower(),
                    pest.get("hindi_name", "").lower(),
                    pest["type"].lower().split()[0],
                ]
            ):
                matched_pests.append(pest)

        display_pests = matched_pests if matched_pests else pests

        for pest in display_pests:
            lines.append(f"🔴 {pest['pest_name']} ({pest.get('hindi_name', '')})")
            lines.append(f"   Type: {pest['type']}")
            lines.append(f"   Stage: {pest['affected_stage']}")
            lines.append(f"   Symptoms: {pest['symptoms'][:200]}...")
            lines.append(f"   Conditions: {pest['favorable_conditions']}")
            lines.append("   Prevention:")
            for p in pest.get("prevention", [])[:3]:
                lines.append(f"     • {p}")
            lines.append("   🌿 Organic Remedies:")
            for o in pest.get("organic_remedies", [])[:2]:
                lines.append(f"     • {o}")
            lines.append("   💊 Chemical Remedies:")
            for c in pest.get("chemical_remedies", [])[:2]:
                lines.append(f"     • {c}")
            lines.append(f"   ⚠️  Threshold: {pest.get('economic_threshold', 'N/A')}")
            lines.append("")

        # Add organic inputs reference
        organic_inputs = general.get("organic_inputs", {})
        if organic_inputs:
            lines.append("🌱 Useful Organic Inputs:")
            for name, desc in list(organic_inputs.items())[:3]:
                lines.append(f"  • {name}: {desc[:100]}...")

        return "\n".join(lines)

    except FileNotFoundError:
        return "Pest data file not found. Please run the data ingestion script."
    except Exception as e:
        logger.error(f"Pest tool error: {e}")
        return f"Error fetching pest data: {str(e)}"


def get_pest_tool() -> Tool:
    """Return a LangChain Tool for pest and disease queries."""
    return Tool(
        name="pest_disease_control",
        description=(
            "Get pest and disease information for crops including symptoms, prevention, "
            "organic remedies, and chemical treatments. "
            "Supports crops: Wheat, Rice, Tomato, Cotton. "
            "Use when farmer asks about: insects, pests, diseases, yellow leaves, "
            "crop damage, fungal attack, bollworm, rust, blight, etc."
        ),
        func=get_pest_info,
    )
