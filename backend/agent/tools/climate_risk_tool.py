"""
Climate Risk Tool — Early Warning System for Agricultural Threats
Predicts Drought, Flood, Heat Stress, Frost, Lodging, and Fungal Risk
using real-time weather, satellite soil moisture, and crop phenology.
"""
import logging
from typing import Optional

from langchain_core.tools import Tool
from services.climate_risk_service import get_climate_risk_service

logger = logging.getLogger(__name__)


def get_climate_risk_report(query: str, default_location: str = "Delhi") -> str:
    """
    Evaluates real-time climate risks (drought, flood, heat stress, frost, lodging)
    for a specified crop and location.
    """
    try:
        svc = get_climate_risk_service()

        # Extract crop from query
        q_lower = query.lower()
        if any(w in q_lower for w in ["wheat", "gehu", "gehun", "गेहूं", "ਕਣਕ"]):
            crop = "wheat"
        elif any(w in q_lower for w in ["rice", "paddy", "dhan", "धान", "चावल", "ਝੋਨਾ"]):
            crop = "rice"
        elif any(w in q_lower for w in ["tomato", "tamatar", "टमाटर"]):
            crop = "tomato"
        elif any(w in q_lower for w in ["potato", "aloo", "आलू"]):
            crop = "potato"
        elif any(w in q_lower for w in ["mustard", "sarson", "सरसों"]):
            crop = "mustard"
        elif any(w in q_lower for w in ["maize", "makka", "मक्का"]):
            crop = "maize"
        elif any(w in q_lower for w in ["cotton", "kapas", "कपास"]):
            crop = "cotton"
        else:
            crop = "wheat"

        # Extract location if mentioned
        location = default_location
        for city in ["Delhi", "Mumbai", "Pune", "Lucknow", "Jaipur", "Ludhiana", "Chandigarh", "Bhopal", "Patna", "Nashik"]:
            if city.lower() in q_lower:
                location = city
                break

        res = svc.assess_crop_climate_risk(location=location, crop=crop)

        lines = [
            f"🛡️ Climate Risk & Early Warning Assessment for **{res['crop']}** at **{res['location']}** ({res['state']}):",
            f"🟢 Source: Real-Time Open-Meteo Satellite & Meteorological Model",
            f"📊 Overall Climate Vulnerability Index (CVI): **{res['climate_vulnerability_index']}/100** ({res['cvi_status']} {res['cvi_icon']})",
            f"🌱 Analyzed Growth Stage: **{res['growth_stage']}**",
            "",
            "💧 **Satellite Root Zone Soil Moisture Profile:**",
            f"• Surface (0–1cm): {res['satellite_soil_moisture']['surface_0_to_1cm']} m³/m³",
            f"• Active Root Zone (3–9cm): {res['satellite_soil_moisture']['active_root_3_to_9cm']} m³/m³",
            f"• Subsoil (9–27cm): {res['satellite_soil_moisture']['deep_root_9_to_27cm']} m³/m³",
            f"• Soil Moisture Status: **{res['satellite_soil_moisture']['status']}** (Wilting threshold: {res['satellite_soil_moisture']['wilting_threshold']} m³/m³)",
            "",
            "⚠️ **Climate Threat Evaluation:**",
        ]

        scores = res.get("risk_scores", {})
        for risk_key, info in scores.items():
            r_name = risk_key.replace("_", " ").title()
            lines.append(f"• {info['icon']} **{r_name}**: {info['score']}% ({info['level']})")

        alerts = res.get("active_alerts", [])
        if alerts:
            lines.append("\n🚨 **Active Early Warnings:**")
            for a in alerts:
                lines.append(f"• {a['icon']} **{a['type']} ({a['severity']})**: {a['message']}")

        actions = res.get("preventive_actions", [])
        if actions:
            lines.append("\n💡 **Prescribed Preventive Actions (ICAR / Agronomy Guidance):**")
            for act in actions:
                lines.append(act)

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Climate risk tool error: {e}")
        return f"Unable to assess climate risk: {str(e)}"


def get_climate_risk_tool(default_location: str = "Delhi") -> Tool:
    """Return LangChain tool for climate risk predictions and early warnings."""
    return Tool(
        name="climate_risk_early_warning",
        description=(
            "Assess climate risks and early warnings for crops: drought risk, flood risk, "
            "heat stress warnings, frost alerts, heavy rainfall lodging, and fungal disease risk. "
            "Uses real-time weather forecasts, historical rainfall, and satellite root zone soil moisture."
        ),
        func=lambda q: get_climate_risk_report(q, default_location),
    )

