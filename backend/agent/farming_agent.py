"""
KrishiRakshak Farming Agent.
High-performance agricultural advice agent with multi-language support (English and Hindi).
Uses direct domain-tool routing + RAG retrieval + Groq LLM for sub-second responses.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure backend directory is in sys.path for robust standalone/Streamlit execution
_BACKEND_DIR = str(Path(__file__).parent.parent.resolve())
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm_client import get_llm
from agent.rag_pipeline import similarity_search
from agent.tools.weather_tool import get_weather_tool
from agent.tools.crop_tool import get_crop_tool
from agent.tools.pest_tool import get_pest_tool
from agent.tools.market_tool import get_market_tool
from agent.tools.disease_detection_tool import create_disease_tool
from agent.tools.climate_risk_tool import get_climate_risk_tool

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_EN = """You are KrishiRakshak, an expert AI farming assistant for Indian farmers.
You provide practical, accurate, and actionable agricultural advice.
Always be empathetic and use simple language that farmers can understand.
Respond in clear, simple English.
When you have enough information, give a comprehensive helpful answer with practical recommendations.
Always mention the data sources you used."""

SYSTEM_PROMPT_HI = """आप कृषि रक्षक हैं, भारतीय किसानों के लिए एक विशेषज्ञ AI सहायक।
आप व्यावहारिक, सटीक और कार्रवाई योग्य कृषि सलाह देते हैं।
हमेशा सरल ग्रामीण हिंदी भाषा में जवाब दें जो किसान समझ सकें।
जब पर्याप्त जानकारी मिल जाए, तो व्यापक और उपयोगी उत्तर दें।"""

SYSTEM_PROMPT_PA = """ਤੁਸੀਂ ਕ੍ਰਿਸ਼ੀ ਰੱਖਿਅਕ (KrishiRakshak) ਹੋ, ਭਾਰਤੀ ਕਿਸਾਨਾਂ ਲਈ ਇੱਕ ਮਾਹਰ AI ਖੇਤੀਬਾੜੀ ਸਹਾਇਕ।
ਤੁਸੀਂ ਵਿਹਾਰਕ, ਸਟੀਕ ਅਤੇ ਲਾਭਦਾਇਕ ਖੇਤੀ ਸਲਾਹ ਦਿੰਦੇ ਹੋ।
ਹਮੇਸ਼ਾ ਸਰਲ, ਸਪੱਸ਼ਟ ਪੰਜਾਬੀ ਬੋਲੀ (ਗੁਰਮੁਖੀ ਲਿਪੀ) ਵਿੱਚ ਜਵਾਬ ਦਿਓ ਤਾਂ ਜੋ ਕਿਸਾਨ ਆਸਾਨੀ ਨਾਲ ਸਮਝ ਸਕਣ।
ਜਦੋਂ ਬਿਮਾਰੀ ਜਾਂ ਕੀੜੇ ਬਾਰੇ ਪੁੱਛਿਆ ਜਾਵੇ, ਤਾਂ ਲੱਛਣ, ਕਾਰਨ, ਦੇਸੀ/ਜੈਵਿਕ ਇਲਾਜ ਅਤੇ ਰਸਾਇਣਕ ਰੋਕਥਾਮ ਸਪੱਸ਼ਟ ਦੱਸੋ।
ਹਮੇਸ਼ਾ ਭਰੋਸੇਯੋਗ ਖੇਤੀਬਾੜੀ ਜਾਣਕਾਰੀ ਅਤੇ ਡੇਟਾ ਸਰੋਤਾਂ ਦਾ ਹਵਾਲਾ ਦਿਓ।"""


def _make_rag_tool() -> Tool:
    """Create a RAG knowledge base search tool."""
    def rag_search(query: str) -> str:
        try:
            docs = similarity_search(query, k=3)
            if not docs:
                return "No relevant information found in the knowledge base."
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Agricultural Knowledge Base")
                results.append(f"[Source {i}: {source}]\n{doc.page_content}")
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return f"Knowledge base search failed: {str(e)}"

    return Tool(
        name="agricultural_knowledge_base",
        description="Search the agricultural knowledge base for general farming information and expert advice.",
        func=rag_search,
    )


def run_agent(
    message: str,
    language: str = "English",
    location: str = "Delhi",
) -> Dict[str, Any]:
    """
    Run the KrishiRakshak agent and return a structured response.
    Routes queries to domain tools with sub-second synthesis.
    """
    llm = None
    try:
        llm = get_llm()
    except Exception as ex:
        logger.warning(f"LLM client setup notice: {ex}")
    lang_clean = language.lower()
    if lang_clean in ("hindi", "hi"):
        system_prompt = SYSTEM_PROMPT_HI
    elif lang_clean in ("punjabi", "pa"):
        system_prompt = SYSTEM_PROMPT_PA
    else:
        system_prompt = SYSTEM_PROMPT_EN

    msg_lower = message.lower()

    # Domain tools
    tool_instances = {
        "weather_forecast": get_weather_tool(location),
        "crop_recommendation": get_crop_tool(),
        "pest_disease_control": get_pest_tool(),
        "disease_info": create_disease_tool(),
        "mandi_price_checker": get_market_tool(),
        "climate_risk_early_warning": get_climate_risk_tool(location),
    }

    keyword_map = {
        "climate_risk_early_warning": [
            "risk", "alert", "warning", "drought", "flood", "heat stress", "frost", "cold wave",
            "heavy rain", "waterlogging", "soil moisture", "lodging", "climate risk", "early warning",
            "sukha", "badh", "pala", "chilling",
            "जोखिम", "चेतावनी", "अलर्ट", "सूखा", "बाढ़", "पाला", "तुषार", "लू", "अतिवृष्टि", "ओलावृष्टि", "नमी",
            "ਖ਼ਤਰਾ", "ਚੇਤਾਵਨੀ", "ਸੋਕਾ", "ਹੜ੍ਹ", "ਪਾਲਾ", "ਤਪਸ਼", "ਲੂ", "ਸਿੱਲ੍ਹ"
        ],
        "weather_forecast": [
            "weather", "rain", "temperature", "forecast", "mausam", "barish",
            "humidity", "wind", "storm", "climate", "hawa", "thand", "garmi", "advisory",
            "मौसम", "बारिश", "बरसात", "तापमान", "हवा", "ठंड", "गर्मी", "सलाह",
            "ਮੌਸਮ", "ਮੀਂਹ", "ਤਾਪਮਾਨ", "ਹਵਾ", "ਬਾਰਿਸ਼", "ਸਲਾਹ"
        ],
        "crop_recommendation": [
            "crop", "season", "kharif", "rabi", "zaid", "sow", "plant", "fasal",
            "beej", "variety", "yield", "gehun", "wheat", "rice", "chawal", "dhan",
            "makka", "corn", "cotton", "kapas", "mustard", "sarson", "potato", "aloo",
            "sowing", "harvest",
            "फसल", "खरीफ", "रबी", "जायद", "बोने", "बीज", "पैदावार", "किस्म", "गेहूं",
            "धान", "चावल", "मक्का", "कपास", "सरसों", "आलू",
            "ਫਸਲ", "ਕਣਕ", "ਝੋਨਾ", "ਬੀਜ", "ਖਰੀਫ", "ਰਬੀ", "ਮੱਕੀ", "ਕਪਾਹ", "ਆਲੂ"
        ],
        "pest_disease_control": [
            "pest", "insect", "keet", "keda", "illia", "sundhi", "spray",
            "pesticide", "neem", "caterpillar", "aphid", "borer",
            "कीट", "कीड़ा", "रोकथाम", "दवा", "कीटनाशक", "नीम", "स्प्रे",
            "ਕੀੜੇ", "ਸੁੰਡੀ", "ਸਪਰੇਅ", "ਕੀਟਨਾਸ਼ਕ", "ਰੋਕਥਾਮ", "ਤੇਲਾ"
        ],
        "disease_info": [
            "disease", "leaf", "symptom", "treatment", "bimari", "rog", "blight",
            "rust", "spot", "yellow", "curl", "fungus", "wilt", "rot",
            "रोग", "बीमारी", "झुलसा", "जंग", "धब्बा", "पत्ती", "फफूंद",
            "ਬਿਮਾਰੀ", "ਪੱਤੇ", "ਪੱਤਿਆਂ", "ਪੀਲੇ", "ਧੱਬੇ", "ਕੁੰਗੀ", "ਝੁਲਸਾ", "ਫੰਗਸ", "ਰੋਗ", "ਟਮਾਟਰ"
        ],
        "mandi_price_checker": [
            "price", "mandi", "rate", "bhav", "daam", "market", "sell", "cost",
            "msp", "quintal", "rupee", "rs", "bazar", "kharid",
            "मंडी", "भाव", "दाम", "रेट", "मूल्य", "बाजार", "टमाटर", "क्विंटल", "खरीद", "बेचने",
            "ਮੰਡੀ", "ਭਾਅ", "ਰੇਟ", "ਕੀਮਤ", "ਕੁਇੰਟਲ"
        ],
    }

    tool_outputs = []
    sources = []
    agent_steps = []

    # Check matching domain tools
    for tool_name, keywords in keyword_map.items():
        if any(kw in msg_lower for kw in keywords):
            tool = tool_instances.get(tool_name)
            if tool:
                try:
                    output = tool.func(message)
                    tool_outputs.append(f"[{tool_name}]\n{output}")
                    sources.append(tool_name)
                    agent_steps.append({
                        "tool": tool_name,
                        "input": message,
                        "output": str(output)[:300],
                    })
                except Exception as e:
                    logger.warning(f"Tool {tool_name} error: {e}")

    # If no specific domain tool matched, or query is broad, supplement with RAG
    if not tool_outputs or any(kw in msg_lower for kw in ["soil", "fertilizer", "khad", "organic", "water", "irrigation"]):
        try:
            rag_tool = _make_rag_tool()
            rag_output = rag_tool.func(message)
            if "No relevant information found" not in rag_output and "failed" not in rag_output:
                tool_outputs.append(f"[agricultural_knowledge_base]\n{rag_output}")
                sources.append("agricultural_knowledge_base")
                agent_steps.append({
                    "tool": "agricultural_knowledge_base",
                    "input": message,
                    "output": str(rag_output)[:300],
                })
        except Exception as e:
            logger.warning(f"RAG search warning: {e}")

    # Build prompt for LLM synthesis
    context = "\n\n".join(tool_outputs) if tool_outputs else "No specific tool data found for this question."
    augmented_message = f"{message} (Farmer's Location: {location})"

    lang_instruction = "Answer in clear, polite, structured English with practical recommendations and an action checklist."
    if lang_clean in ("hindi", "hi"):
        lang_instruction = "कृपया उत्तर पूरी तरह से शुद्ध, सरल और आदरणीय हिंदी (Devanagari script) में दें। व्यावहारिक सुझाव, रोकथाम के उपाय और स्पष्ट बुलेट पॉइंट्स या तालिका का उपयोग करें।"
    elif lang_clean in ("punjabi", "pa"):
        lang_instruction = "ਕਿਰਪਾ ਕਰਕੇ ਪੂਰਾ ਜਵਾਬ ਸਰਲ ਅਤੇ ਸਪੱਸ਼ਟ ਪੰਜਾਬੀ (Gurmukhi script) ਵਿੱਚ ਦਿਓ। ਕਿਸਾਨ ਵੀਰਾਂ ਲਈ ਵਿਹਾਰਕ ਸਲਾਹ ਅਤੇ ਬੁਲੇਟ ਪੁਆਇੰਟਾਂ ਦੀ ਵਰਤੋਂ ਕਰੋ।"

    full_prompt = (
        f"{system_prompt}\n\n"
        f"--- REAL-TIME AGRICULTURAL DATA ---\n"
        f"{context}\n\n"
        f"--- FARMER QUESTION ---\n"
        f"{augmented_message}\n\n"
        f"Instructions:\n"
        f"1. {lang_instruction}\n"
        f"2. Use the provided real-time agricultural data accurately.\n"
        f"3. Always mention the data source (e.g., Open-Meteo, Agmarknet) and keep the advice actionable for farmers."
    )

    if llm is not None:
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=full_prompt),
            ])
            return {
                "response": response.content,
                "sources": sources,
                "agent_steps": agent_steps,
            }
        except Exception as e:
            logger.error(f"LLM synthesis error: {e}")

    # Polished Fallback: return verified domain tool telemetry in a clear, formatted presentation
    if tool_outputs:
        import re
        cleaned_outputs = []
        for out in tool_outputs:
            clean_str = re.sub(r"^\[[a-zA-Z0-9_]+\]\s*", "", out.strip())
            if clean_str:
                cleaned_outputs.append(clean_str)

        retrieved_data = "\n\n---\n\n".join(cleaned_outputs)

        if lang_clean in ("hindi", "hi"):
            header = "🌾 **सत्यापित कृषि परामर्श एवं मौसम बुलेटिन (रियल-टाइम डेटा):**"
            footer_tip = "💡 *सुझाव: अधिक विस्तृत AI बातचीत और व्यक्तिगत सवालों के लिए बायीं ओर साइडबार में अपनी Groq API Key दर्ज करें।*"
        elif lang_clean in ("punjabi", "pa"):
            header = "🌾 **ਪ੍ਰਮਾਣਿਤ ਖੇਤੀਬਾੜੀ ਸਲਾਹ ਅਤੇ ਮੌਸਮ ਜਾਣਕਾਰੀ (ਰੀਅਲ-ਟਾਈਮ ਡੇਟਾ):**"
            footer_tip = "💡 *ਸੁਝਾਅ: ਵਿਸਤ੍ਰਿਤ AI ਜਵਾਬਾਂ ਲਈ ਖੱਬੇ ਪਾਸੇ ਸਾਈਡਬਾਰ ਵਿੱਚ ਆਪਣੀ Groq API Key ਦਰਜ ਕਰੋ।*"
        else:
            header = "🌾 **Verified Agricultural Advisory & Real-Time Intelligence:**"
            footer_tip = "💡 *Tip: For interactive conversational answers and custom follow-ups, activate your Groq API key in the left sidebar or in Streamlit Cloud Secrets.*"

        return {
            "response": f"{header}\n\n{retrieved_data}\n\n---\n{footer_tip}",
            "sources": sources,
            "agent_steps": agent_steps,
        }

    return {
        "response": (
            "⚠️ **Groq API Key Required for AI Synthesis**\n\n"
            "To ask custom free-form questions, please enter your free Groq API key in the left sidebar.\n\n"
            "👉 [Get a free Groq API Key at console.groq.com/keys](https://console.groq.com/keys)\n\n"
            "*If running on Streamlit Cloud, add `LLM_API_KEY = \"gsk_...\"` in App Settings → Secrets.*"
        ),
        "sources": sources,
        "agent_steps": agent_steps,
    }
