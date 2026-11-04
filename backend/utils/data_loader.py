"""
DataLoader — reads all JSON data files, chunks them into Documents,
and stores them in ChromaDB via the RAG pipeline.
"""
import json
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from agent.rag_pipeline import add_documents, get_vectorstore

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class DataLoader:
    """Loads all farming data files into the ChromaDB vectorstore."""

    def load_all(self) -> None:
        """Load all data files. Skip if collection already has documents."""
        vs = get_vectorstore()
        existing = vs._collection.count()
        if existing > 0:
            logger.info(
                f"ChromaDB already contains {existing} documents. Skipping ingestion."
            )
            return

        logger.info("Ingesting data files into ChromaDB...")
        all_docs: List[Document] = []
        all_docs.extend(self._load_weather())
        all_docs.extend(self._load_crops())
        all_docs.extend(self._load_pests())
        all_docs.extend(self._load_market())
        all_docs.extend(self._load_soil())

        if all_docs:
            add_documents(all_docs)
            logger.info(f"Ingested {len(all_docs)} document chunks into ChromaDB.")
        else:
            logger.warning("No documents were loaded from data files.")

    # ------------------------------------------------------------------ #
    # Private loaders
    # ------------------------------------------------------------------ #

    def _load_weather(self) -> List[Document]:
        path = DATA_DIR / "weather_data.json"
        docs = []
        try:
            with open(path) as f:
                data = json.load(f)
            for city, city_data in data.get("cities", {}).items():
                for forecast in city_data.get("forecasts", []):
                    text = (
                        f"Weather forecast for {city} ({city_data.get('state', '')}) "
                        f"on {forecast['date']} ({forecast['day']}): "
                        f"Condition: {forecast['condition']}. "
                        f"Temperature: {forecast['temperature']['min']}–"
                        f"{forecast['temperature']['max']}°C. "
                        f"Humidity: {forecast['humidity']}%. "
                        f"Rainfall: {forecast['rainfall']} mm. "
                        f"Wind: {forecast['wind_speed']} km/h. "
                        f"Farming advisory: {forecast['advisory']}"
                    )
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": "Weather Data",
                            "city": city,
                            "date": forecast["date"],
                            "type": "weather",
                        },
                    ))
                # City-level crop impact note
                if city_data.get("crop_impact"):
                    docs.append(Document(
                        page_content=(
                            f"Agricultural crop impact summary for {city}: "
                            f"{city_data['crop_impact']}"
                        ),
                        metadata={"source": "Weather Data", "city": city, "type": "weather_summary"},
                    ))
            logger.info(f"Loaded {len(docs)} weather documents.")
        except Exception as e:
            logger.error(f"Failed to load weather data: {e}")
        return docs

    def _load_crops(self) -> List[Document]:
        path = DATA_DIR / "crop_calendar.json"
        docs = []
        try:
            with open(path) as f:
                data = json.load(f)
            for season, season_data in data.get("seasons", {}).items():
                for crop in season_data.get("crops", []):
                    text = (
                        f"{season} crop: {crop['name']} (Hindi: {crop['hindi_name']}). "
                        f"Sowing months: {', '.join(season_data.get('sowing_months', []))}. "
                        f"Harvest months: {', '.join(season_data.get('harvesting_months', []))}. "
                        f"Suitable soil: {', '.join(crop['soil_types'])}. "
                        f"Soil pH: {crop['soil_ph']}. "
                        f"Water requirement: {crop['water_requirement']}. "
                        f"Growing period: {crop['growing_period_days']} days. "
                        f"Expected yield: {crop['expected_yield_per_acre']} per acre. "
                        f"Suitable regions: {', '.join(crop['suitable_regions'])}. "
                        f"Key varieties: {', '.join(crop['key_varieties'])}. "
                        f"Fertilizer: {crop['fertilizer_requirement']}. "
                        f"Key diseases: {', '.join(crop.get('key_diseases', []))}. "
                        f"Planting tip: {crop['planting_tips']}. "
                        f"Advisory: {crop['advisory']}"
                    )
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": "Crop Calendar",
                            "crop": crop["name"],
                            "season": season,
                            "type": "crop",
                        },
                    ))
            # Soil-crop mapping
            for soil, crops in data.get("soil_crop_mapping", {}).items():
                docs.append(Document(
                    page_content=(
                        f"Soil type {soil} is suitable for the following crops: "
                        f"{', '.join(crops)}."
                    ),
                    metadata={"source": "Crop Calendar", "type": "soil_crop_mapping"},
                ))
            logger.info(f"Loaded {len(docs)} crop documents.")
        except Exception as e:
            logger.error(f"Failed to load crop data: {e}")
        return docs

    def _load_pests(self) -> List[Document]:
        path = DATA_DIR / "pest_control.json"
        docs = []
        try:
            with open(path) as f:
                data = json.load(f)
            for crop, pests in data.get("pests_by_crop", {}).items():
                for pest in pests:
                    text = (
                        f"Pest/Disease for {crop}: {pest['pest_name']} "
                        f"(Hindi: {pest.get('hindi_name', 'N/A')}). "
                        f"Type: {pest['type']}. "
                        f"Affected stage: {pest['affected_stage']}. "
                        f"Symptoms: {pest['symptoms']} "
                        f"Favorable conditions: {pest['favorable_conditions']}. "
                        f"Prevention: {'; '.join(pest.get('prevention', []))}. "
                        f"Organic remedies: {'; '.join(pest.get('organic_remedies', []))}. "
                        f"Chemical remedies: {'; '.join(pest.get('chemical_remedies', []))}. "
                        f"Economic threshold: {pest.get('economic_threshold', 'N/A')}."
                    )
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": "Pest Control Guide",
                            "crop": crop,
                            "pest": pest["pest_name"],
                            "type": "pest",
                        },
                    ))
            # General IPM principles
            ipm = data.get("general_pest_management", {}).get("IPM_principles", [])
            if ipm:
                docs.append(Document(
                    page_content=(
                        "Integrated Pest Management (IPM) principles: "
                        + "; ".join(ipm)
                    ),
                    metadata={"source": "Pest Control Guide", "type": "ipm"},
                ))
            logger.info(f"Loaded {len(docs)} pest documents.")
        except Exception as e:
            logger.error(f"Failed to load pest data: {e}")
        return docs

    def _load_market(self) -> List[Document]:
        path = DATA_DIR / "mandi_prices.json"
        docs = []
        try:
            with open(path) as f:
                data = json.load(f)
            for market, market_data in data.get("markets", {}).items():
                for commodity, price in market_data.get("commodities", {}).items():
                    text = (
                        f"Mandi price for {commodity} at {market} "
                        f"({market_data.get('state', '')}) on {data.get('date', 'N/A')}: "
                        f"Minimum ₹{price['min']}/quintal, "
                        f"Maximum ₹{price['max']}/quintal, "
                        f"Modal price ₹{price['modal']}/quintal. "
                        f"Price trend: {price['trend']}. "
                        f"Remarks: {price['remarks']}."
                    )
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": "Mandi Price Data",
                            "market": market,
                            "commodity": commodity,
                            "type": "market",
                        },
                    ))
            # MSP data
            msps = data.get("msps", {})
            msp_lines = [
                f"{crop}: ₹{price}/quintal"
                for crop, price in msps.get("crops", {}).items()
            ]
            if msp_lines:
                docs.append(Document(
                    page_content=(
                        "Government Minimum Support Prices (MSP) for 2023-24: "
                        + "; ".join(msp_lines)
                        + f". {msps.get('note', '')}"
                    ),
                    metadata={"source": "MSP Data", "type": "msp"},
                ))
            logger.info(f"Loaded {len(docs)} market documents.")
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
        return docs

    def _load_soil(self) -> List[Document]:
        path = DATA_DIR / "soil_conditions.json"
        docs = []
        try:
            with open(path) as f:
                data = json.load(f)
            for region in data.get("regions", []):
                for profile in region.get("soil_profiles", []):
                    nutrients_text = "; ".join(
                        f"{k}: {v}"
                        for k, v in profile.get("major_nutrients", {}).items()
                    )
                    text = (
                        f"Soil information for {region['region_name']} "
                        f"(States: {', '.join(region['states'])}): "
                        f"Soil type: {profile['soil_type']} ({profile.get('local_name', '')}). "
                        f"Texture: {profile['texture']}. "
                        f"pH range: {profile['ph_range']}. "
                        f"Organic matter: {profile['organic_matter']}. "
                        f"Nutrients: {nutrients_text}. "
                        f"Water holding capacity: {profile['water_holding_capacity']}. "
                        f"Suitable crops: {', '.join(profile['suitable_crops'])}. "
                        f"Limitations: {'; '.join(profile.get('limitations', []))}. "
                        f"Recommendations: {'; '.join(profile.get('recommendations', []))}."
                    )
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": "Soil Conditions Data",
                            "region": region["region_name"],
                            "soil_type": profile["soil_type"],
                            "type": "soil",
                        },
                    ))
            # Soil health indicators
            health = data.get("soil_health_indicators", {})
            if health.get("soil_testing_advice"):
                docs.append(Document(
                    page_content=f"Soil testing advice: {health['soil_testing_advice']}",
                    metadata={"source": "Soil Health Guide", "type": "soil_health"},
                ))
            logger.info(f"Loaded {len(docs)} soil documents.")
        except Exception as e:
            logger.error(f"Failed to load soil data: {e}")
        return docs
