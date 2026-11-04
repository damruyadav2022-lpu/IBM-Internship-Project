"""
Market Tool — queries mandi_prices.json for current commodity prices.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import Tool
from services.mandi_service import get_mandi_service

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "mandi_prices.json"

_market_data: Optional[dict] = None


def _load_market_data() -> dict:
    global _market_data
    if _market_data is None:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            _market_data = json.load(f)
    return _market_data


# Commodity name aliases (Hindi/common names → canonical name)
COMMODITY_ALIASES = {
    "aloo": "Potato",
    "tamatar": "Tomato",
    "pyaz": "Onion",
    "gehu": "Wheat",
    "gehun": "Wheat",
    "chawal": "Rice (Basmati)",
    "sarson": "Mustard",
    "lehsun": "Garlic",
    "matar": "Peas",
    "makka": "Maize",
    "chana": "Chickpea",
    "dhaniya": "Coriander (Dhaniya)",
    "jeera": "Cumin (Jeera)",
    "hari mirch": "Green Chilli",
    "gobhi": "Cauliflower",
    "angoor": "Grapes",
    "anaar": "Pomegranate",
    "kela": "Banana",
    "moongphali": "Groundnut",
    "ganna": "Sugarcane",
    "baingan": "Brinjal",
    "arhar": "Arhar Dal",
    "moong": "Moong Dal",
    "cotton": "Cotton (Medium Staple)",
    "kapas": "Cotton (Medium Staple)",
    "soyabean": "Soybean",
    "jowar": "Jowar (Hybrid)",
    "bajra": "Bajra",
    # Hindi Devanagari aliases
    "टमाटर": "Tomato",
    "आलू": "Potato",
    "प्याज": "Onion",
    "गेहूं": "Wheat",
    "गेहुं": "Wheat",
    "चावल": "Rice (Basmati)",
    "धान": "Rice (Basmati)",
    "सरसों": "Mustard",
    "लहसुन": "Garlic",
    "मटर": "Peas",
    "मक्का": "Maize",
    "चना": "Chickpea",
    "धनिया": "Coriander (Dhaniya)",
    "जीरा": "Cumin (Jeera)",
    "मिर्च": "Green Chilli",
    "गोभी": "Cauliflower",
    "अंगूर": "Grapes",
    "अनार": "Pomegranate",
    "केला": "Banana",
    "मूंगफली": "Groundnut",
    "गन्ना": "Sugarcane",
    "बैंगन": "Brinjal",
    "कपास": "Cotton (Medium Staple)",
    "सोयाबीन": "Soybean",
}

HINDI_MARKET_MAP = {
    "दिल्ली": "Azadpur Mandi",
    "मुंबई": "Vashi APMC",
    "पुणे": "Gultekdi Market Yard",
    "लखनऊ": "Dubagga Mandi",
    "जयपुर": "Muhana Mandi",
}


def _find_commodity(query: str, market_data: dict) -> Optional[str]:
    """Find commodity name in query using aliases and direct match."""
    q = query.lower()
    # Check aliases first
    for alias, canonical in COMMODITY_ALIASES.items():
        if alias in query or alias in q:
            return canonical
    # Direct case-insensitive check across all markets
    for market_info in market_data.get("markets", {}).values():
        for commodity in market_info.get("commodities", {}):
            if commodity.lower() in q:
                return commodity
    return None


def _find_market(query: str, markets: dict) -> Optional[str]:
    """Find market/city in query, supporting Hindi city names."""
    q = query.lower()
    for h_city, m_part in HINDI_MARKET_MAP.items():
        if h_city in query:
            for market_name in markets:
                if m_part in market_name:
                    return market_name

    for market_name in markets:
        # Match on city name (first part before bracket)
        city = market_name.split("(")[0].strip().lower()
        if city in q or any(part.lower() in q for part in market_name.split()):
            return market_name
    return None


def get_market_prices(query: str, location: str = "Delhi") -> str:
    """
    Return real-time live APMC mandi prices for a commodity, with Government of India
    Agmarknet live integration and MSP 2024-25 benchmarks.
    """
    try:
        svc = get_mandi_service()
        return svc.get_commodity_price_report(query, location=location)
    except Exception as e:
        logger.warning(f"Live MandiService notice: {e}, falling back to offline records.")
        try:
            data = _load_market_data()
            markets = data.get("markets", {})
            msps = data.get("msps", {})

            commodity = _find_commodity(query, data)
            market_filter = _find_market(query, markets)

            lines = [
                f"📊 Mandi Price Report — {data.get('date', 'N/A')}",
                f"Unit: {data.get('unit', 'INR/quintal')}",
                "",
            ]

            if commodity:
                lines.append(f"💰 Prices for: {commodity}")
                lines.append("")
                found_any = False
                for market_name, market_info in markets.items():
                    if market_filter and market_name != market_filter:
                        continue
                    commodities = market_info.get("commodities", {})
                    price_info = commodities.get(commodity)
                    if not price_info:
                        for c_name, c_info in commodities.items():
                            if commodity.lower() in c_name.lower():
                                price_info = c_info
                                commodity = c_name
                                break
                    if price_info:
                        found_any = True
                        trend_icon = (
                            "📈" if price_info["trend"] == "rising"
                            else "📉" if price_info["trend"] == "falling"
                            else "➡️"
                        )
                        lines.append(f"🏪 {market_name} ({market_info.get('state', '')})")
                        lines.append(
                            f"   Min: ₹{price_info['min']:,}  |  "
                            f"Max: ₹{price_info['max']:,}  |  "
                            f"Modal: ₹{price_info['modal']:,} per quintal"
                        )
                        lines.append(
                            f"   Trend: {trend_icon} {price_info['trend'].capitalize()}  |  "
                            f"{price_info['remarks']}"
                        )
                        lines.append("")

                if not found_any:
                    lines.append(
                        f"Price data for '{commodity}' not found in available markets."
                    )

                # Check MSP
                msp_crops = msps.get("crops", {})
                for crop_name, msp_price in msp_crops.items():
                    if commodity.lower() in crop_name.lower():
                        lines.append(
                            f"🏛️  Government MSP for {crop_name}: ₹{msp_price:,}/quintal"
                        )
                        lines.append(
                            "   Tip: If market price is below MSP, sell to government procurement center."
                        )
                        break

            else:
                target_market = market_filter or next(iter(markets))
                market_info = markets.get(target_market, {})
                lines.append(f"🏪 {target_market} — Today's Prices:")
                lines.append("")
                for commodity_name, price_info in list(
                    market_info.get("commodities", {}).items()
                )[:8]:
                    trend_icon = (
                        "📈" if price_info["trend"] == "rising"
                        else "📉" if price_info["trend"] == "falling"
                        else "➡️"
                    )
                    lines.append(
                        f"  {trend_icon} {commodity_name}: "
                        f"₹{price_info['min']:,}–₹{price_info['max']:,} "
                        f"(Modal: ₹{price_info['modal']:,})"
                    )

                lines.append(
                    "\nAsk about a specific commodity for detailed prices across all markets."
                )

            return "\n".join(lines)

        except Exception as ex:
            return f"Error retrieving market data: {str(ex)}"


def get_market_tool() -> Tool:
    """Return a LangChain Tool for mandi price queries."""
    return Tool(
        name="mandi_price_checker",
        description=(
            "Get current mandi (market) prices for agricultural commodities in India. "
            "Returns minimum, maximum, and modal price per quintal in INR. "
            "Also shows price trends and government MSP (Minimum Support Price). "
            "Supports: tomato, onion, potato, wheat, rice, mustard, garlic, "
            "maize, peas, cumin, coriander, groundnut, sugarcane, and more. "
            "Can filter by city: Delhi, Mumbai, Lucknow, Jaipur, Pune."
        ),
        func=get_market_prices,
    )
