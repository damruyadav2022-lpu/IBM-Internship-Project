"""
Soil Health & PAU Fertilizer Tool — Evaluates Soil Health Card values
and generates PAU / ICAR calibrated fertilizer recommendations.
"""
import logging
from typing import Optional

from langchain_core.tools import Tool
from services.soil_health_service import get_soil_health_service

logger = logging.getLogger(__name__)


def get_soil_fertilizer_advice(query: str, default_location: str = "Ludhiana") -> str:
    """
    Evaluates soil health and generates PAU/ICAR fertilizer recommendations
    for Wheat, Paddy, Potato, Mustard, Maize, etc.
    """
    try:
        svc = get_soil_health_service()
        q_lower = query.lower()

        # Identify crop
        if any(w in q_lower for w in ["wheat", "gehu", "gehun", "गेहूं", "ਕਣਕ"]):
            crop = "wheat"
        elif any(w in q_lower for w in ["rice", "paddy", "dhan", "धान", "ਝੋਨਾ"]):
            crop = "rice"
        elif any(w in q_lower for w in ["potato", "aloo", "ਆਲੂ", "आलू"]):
            crop = "potato"
        elif any(w in q_lower for w in ["mustard", "sarson", "ਸਰ੍ਹੋਂ", "सरसों"]):
            crop = "mustard"
        else:
            crop = "wheat"

        # Determine language
        if any(w in q_lower for w in ["ਪੰਜਾਬੀ", "ਕਣਕ", "ਝੋਨਾ", "ਖਾਦ"]):
            language = "Punjabi"
        elif any(w in q_lower for w in ["हिंदी", "गेहूं", "धान", "यूरिया"]):
            language = "Hindi"
        else:
            language = "English"

        # Baseline Ludhiana sample card
        sample = svc.parse_soil_health_card(preloaded_key="Ludhiana Central (PAU Research Zone - Loamy Soil)")
        plan = svc.generate_fertilizer_plan(
            crop=crop,
            soil_values=sample["raw_values"],
            land_acres=1.0,
            language=language,
        )

        lines = [
            f"🧪 **PAU / ICAR Soil Health Card Fertilizer Plan for {plan['crop']}:**",
            f"📍 Calibration: Punjab Agricultural University (PAU) Ludhiana & ICAR",
            f"💰 Estimated Total Cost: ₹{plan['total_cost_inr']:,.0f} per acre",
            "",
            "📋 **Recommended Chemical Fertilizers:**",
        ]
        for f in plan.get("fertilizers", []):
            lines.append(f"• **{f['name']}**: {f['dose_per_acre']} ({f['cost_inr']}) — {f['reason']}")

        lines.append("\n⏳ **Split Application Schedule:**")
        for s in plan.get("splits", []):
            lines.append(f"• **{s['stage']}** ({s['timing']}): {s['fertilizers']} — {s['guidance']}")

        if plan.get("total_saved_inr", 0) > 0:
            lines.append(f"\n💡 **Farmer Cost Saving:** Saved ₹{plan['total_saved_inr']:,.0f} by omitting excess fertilizer.")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Soil tool error: {e}")
        return f"Soil health recommendation temporarily unavailable: {e}"


def get_soil_tool(default_location: str = "Ludhiana") -> Tool:
    return Tool(
        name="soil_health_fertilizer_tool",
        func=lambda q: get_soil_fertilizer_advice(q, default_location=default_location),
        description=(
            "Useful for answering questions about Soil Health Cards, soil test results, "
            "N-P-K nutrient status, organic carbon, pH, and Punjab Agricultural University (PAU) / "
            "ICAR recommended fertilizer doses and split application timing."
        ),
    )
