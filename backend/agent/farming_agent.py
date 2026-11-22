"""
KrishiRakshak Farming Agent.
High-performance agricultural advice agent with multi-language support (11 Indian languages).
Uses direct domain-tool routing + RAG retrieval + Groq LLM for sub-second, structured responses.
Guarantees clean, neat, and clutter-free advice without confusing data dumps.
"""
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from agent.tools.soil_tool import get_soil_tool
from voice_assistant import detect_language, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


SYSTEM_PROMPTS = {
    "English": (
        "You are KrishiRakshak AI, an expert agricultural advisor for Indian farmers. "
        "Provide direct, concise, and beautifully structured advice.\n"
        "STRICT FORMATTING RULES:\n"
        "1. Answer ONLY what the farmer specifically asked. Never dump unrelated crops, weather, or market rates.\n"
        "2. Structure your response into 3 clean, distinct sections:\n"
        "   🎯 **Direct Answer**: 1-2 clear, simple sentences directly answering the question.\n"
        "   📌 **Key Action Points**: Maximum 3 to 4 concise bullet points with practical steps, exact dosages, or dates.\n"
        "   💡 **Farmer Tip**: 1 practical field tip for best results.\n"
        "3. Use simple, conversational language without scientific jargon. Keep total response under 150 words."
    ),
    "Hindi": (
        "आप कृषि रक्षक (KrishiRakshak AI) हैं, भारतीय किसानों के लिए एक विशेषज्ञ कृषि सलाहकार। "
        "हमेशा सीधा, स्पष्ट, स्वच्छ और सुंदर संरचित उत्तर दें।\n"
        "सख्त नियम:\n"
        "1. केवल वही जानकारी दें जो किसान ने पूछी है। अनावश्यक अन्य फसलों, मौसम या मंडी का डेटा कभी न डालें।\n"
        "2. उत्तर को 3 स्पष्ट भागों में प्रस्तुत करें:\n"
        "   🎯 **मुख्य उत्तर**: 1-2 सरल पंक्तियों में सीधा उत्तर।\n"
        "   📌 **कार्य योजना / महत्वपूर्ण बिंदु**: केवल 3-4 संक्षिप्त बुलेट पॉइंट्स (सटीक मात्रा, दवा, किस्म या तारीख)।\n"
        "   💡 **किसान सलाह**: 1 व्यावहारिक खेत सुझाव।\n"
        "3. भाषा एकदम सरल, ग्रामीण और आदरणीय हिंदी रखें। 120-150 शब्दों में संपूर्ण बात स्पष्ट करें।"
    ),
    "Punjabi": (
        "ਤੁਸੀਂ ਕ੍ਰਿਸ਼ੀ ਰੱਖਿਅਕ (KrishiRakshak AI) ਹੋ, ਭਾਰਤੀ ਕਿਸਾਨਾਂ ਲਈ ਮਾਹਰ ਖੇਤੀਬਾੜੀ ਸਲਾਹਕਾਰ। "
        "ਹਮੇਸ਼ਾ ਸਿੱਧਾ, ਸਪੱਸ਼ਟ ਅਤੇ ਸਾਫ਼-ਸੁਥਰਾ ਜਵਾਬ ਦਿਓ।\n"
        "ਸਖ਼ਤ ਨਿਯਮ:\n"
        "1. ਸਿਰਫ਼ ਉਹੀ ਜਾਣਕਾਰੀ ਦਿਓ ਜੋ ਕਿਸਾਨ ਨੇ ਪੁੱਛੀ ਹੈ। ਬੇਲੋੜਾ ਡੇਟਾ ਨਾ ਦਿਓ।\n"
        "2. ਜਵਾਬ ਨੂੰ 3 ਭਾਗਾਂ ਵਿੱਚ ਵੰਡੋ:\n"
        "   🎯 **ਸਿੱਧਾ ਜਵਾਬ**: 1-2 ਲਾਈਨਾਂ ਵਿੱਚ ਮੁੱਖ ਗੱਲ।\n"
        "   📌 **ਮੁੱਖ ਨੁਕਤੇ**: 3-4 ਸੰਖੇਪ ਬੁਲੇਟ ਪੁਆਇੰਟ (ਸਹੀ ਮਾਤਰਾ, ਦਵਾਈ ਜਾਂ ਕਿਸਮ)।\n"
        "   💡 **ਕਿਸਾਨ ਸਲਾਹ**: 1 ਵਿਹਾਰਕ ਖੇਤ ਸੁਝਾਅ।\n"
        "3. ਸਰਲ ਅਤੇ ਸਪੱਸ਼ਟ ਪੰਜਾਬੀ ਬੋਲੀ (ਗੁਰਮੁਖੀ) ਵਰਤੋ।"
    ),
    "Bengali": (
        "আপনি কৃষিরক্ষক (KrishiRakshak AI), ভারতীয় কৃষকদের জন্য একজন বিশেষজ্ঞ কৃষি সহকারী। "
        "সর্বদা স্পষ্ট, সংক্ষিপ্ত এবং পরিচ্ছন্নভাবে কাঠামোগত উত্তর দিন।\n"
        "নিয়ম:\n"
        "1. শুধুমাত্র জিজ্ঞাসিত প্রশ্নের সুনির্দিষ্ট উত্তর দিন।\n"
        "2. উত্তরটি ৩টি স্পষ্ট বিভাগে সাজান: 🎯 **সরাসরি উত্তর**, 📌 **মূল পদক্ষেপ** (সর্বোচ্চ ৩-৪টি বুলেট), 💡 **কৃষকের পরামর্শ**।\n"
        "3. সহজ বাংলা ভাষায় ১২০-১৫০ শব্দের মধ্যে উত্তর রাখুন।"
    ),
    "Telugu": (
        "మీరు కృషిరక్షక్ (KrishiRakshak AI), భారతీయ రైతుల కోసం నిపుణులైన వ్యవసాయ సహాయకుడు. "
        "స్పష్టమైన, సరళమైన మరియు చక్కగా నిర్మాణాత్మకమైన సమాధానం ఇవ్వండి.\n"
        "నియమాలు: 1. అడిగిన ప్రశ్నకు మాత్రమే సమాధానం ఇవ్వండి. 2. మూడు భాగాలుగా చెప్పండి: 🎯 ప్రత్యక్ష సమాధానం, 📌 ముఖ్యమైన అంశాలు, 💡 రైతు చిట్కా. 3. సరళమైన తెలుగు ఉపయోగించండి."
    ),
    "Marathi": (
        "तुम्ही कृषिरक्षक (KrishiRakshak AI) आहात, भारतीय शेतकऱ्यांसाठी एक तज्ज्ञ कृषी सल्लागार. "
        "नेहमी सोप्या, अचूक आणि मुद्देसूद भाषेत मार्गदर्शन करा.\n"
        "नियम: 1. विचारलेल्या प्रश्नाचेच थेट उत्तर द्या. 2. ३ भागांत विभागणी करा: 🎯 मुख्य उत्तर, 📌 कृती योजना (३-४ मुद्दे), 💡 शेतकरी सल्ला."
    ),
    "Tamil": (
        "நீங்கள் கிருஷிரக்ஷக் (KrishiRakshak AI), விவசாயிகளுக்கான சிறந்த வேளாண்மை ஆலோசகர். "
        "எளிய, தெளிவான மற்றும் துல்லியமான பதிலை 3 பிரிவுகளில் வழங்கவும்: 🎯 நேரடி பதில், 📌 முக்கிய குறிப்புகள், 💡 விவசாயி ஆலோசனை."
    ),
    "Gujarati": (
        "તમે કૃષિરક્ષક (KrishiRakshak AI) છો. હંમેશા સરળ, સ્પષ્ટ અને મુદ્દાસર માર્ગદર્શન આપો: 🎯 સીધો જવાબ, 📌 મુખ્ય મુદ્દાઓ, 💡 ખેડૂત સલાહ."
    ),
    "Kannada": (
        "ನೀವು ಕೃಷಿರಕ್ಷಕ್ (KrishiRakshak AI). ಸರಳ ಕನ್ನಡದಲ್ಲಿ ನೇರ ಮತ್ತು ರಚನಾತ್ಮಕ ಸಲಹೆ ನೀಡಿ: 🎯 ನೇರ ಉತ್ತರ, 📌 ಪ್ರಮುಖ ಅಂಶಗಳು, 💡 ರೈತ ಸಲಹೆ."
    ),
    "Malayalam": (
        "നിങ്ങൾ കൃഷിരക്ഷക് (KrishiRakshak AI) ആണ്. ലളിതമായ മലയാളത്തിൽ വ്യക്തമായ നിർദ്ദേശങ്ങൾ നൽകുക: 🎯 നേരിട്ടുള്ള മറുപടി, 📌 പ്രധാന കാര്യങ്ങൾ, 💡 കർഷക ഉപദേശം."
    ),
    "Urdu": (
        "آپ کرشی رکشاک (KrishiRakshak AI) ہیں۔ کسانوں کے لیے صاف، آسان اور باقاعدہ رہنمائی فراہم کریں: 🎯 سیدھا جواب، 📌 اہم نکات، 💡 کسان مشورہ۔"
    ),
}


def _make_rag_tool() -> Tool:
    """Create a RAG knowledge base search tool."""
    def rag_search(query: str) -> str:
        try:
            docs = similarity_search(query, k=2)
            if not docs:
                return "No specific documentation found for this query in the regional manual."
            results = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()
                if len(content) > 300:
                    content = content[:300] + "..."
                results.append(f"• {content}")
            return "\n".join(results)
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return f"Knowledge base search failed: {str(e)}"

    return Tool(
        name="agricultural_knowledge_base",
        description="Search agronomic guidelines and university research manuals.",
        func=rag_search,
    )


def _classify_primary_intent(query: str) -> str:
    """
    Intelligently determine the farmer's primary intent to prevent runaway
    multi-tool collisions (e.g. asking for wheat mandi price should NOT trigger weather or potato data).
    """
    q = query.lower()

    # 1. Mandi / Price Intent (High Priority)
    mandi_keywords = [
        "price", "rate", "bhav", "mandi", "daam", "cost", "sell", "msp", "quintal", "rupee", "bazar", "market",
        "भाव", "दाम", "रेट", "मंडी", "मूल्य", "बाजार", "क्विंटल", "खरीद", "बेचने", "दर", "बाजारभाव",
        "ਭਾਅ", "ਰੇਟ", "ਕੀਮਤ", "ਮੰਡੀ", "কিমਤ", "দর", "দাম", "বাজার", "ధర", "రేటు", "మండి", "விலை", "சந்தை",
        "ભાવ", "કિંમત", "ಬೆಲೆ", "ದರ", "വില", "قیمت", "ریٹ", "منڈی"
    ]
    if any(k in q for k in mandi_keywords):
        return "mandi"

    # 2. Weather & Irrigation Window Intent
    weather_keywords = [
        "weather", "rain", "temperature", "forecast", "mausam", "barish", "humidity", "wind", "storm",
        "मौसम", "बारिश", "बरसात", "तापमान", "हवा", "हवामान", "पाऊस", "गारपीट", "ਮੀਂਹ", "ਤਾਪਮਾਨ", "ਬਰਸਾਤ",
        "আবহাওয়া", "বৃষ্টি", "వాతావరణం", "వర్షం", "வானிலை", "மழை", "હવામાન", "વરસાદ", "ಹವಾಮಾನ", "മഴ", "بارش"
    ]
    if any(k in q for k in weather_keywords):
        return "weather"

    # 3. Pest / Disease / Spray Intent
    pest_keywords = [
        "pest", "insect", "disease", "spray", "pesticide", "fungus", "blight", "rust", "keet", "keda", "bimari", "rog",
        "कीट", "कीड़ा", "रोग", "बीमारी", "झुलसा", "जंग", "धब्बा", "दवा", "कीटनाशक", "स्प्रे", "फफूंद", "कीड", "अळी",
        "ਕੀੜੇ", "ਬਿਮਾਰੀ", "ਸਪਰੇਅ", "ਕੀਟਨਾਸ਼ਕ", "ਕੁੰਗੀ", "ਤੇਲਾ", "ਝੁਲਸਾ", "পোকা", "রোগ", "কীটনাশক", "తెగులు", "పురుగు",
        "பூச்சி", "நோய்", "જીવાત", "ઇયળ", "ರೋಗ", "ಹುಳು", "കീടങ്ങൾ", "کیڑے", "سپرے"
    ]
    if any(k in q for k in pest_keywords):
        return "pest_disease"

    # 4. Soil / Fertilizer / Nutrient Intent
    soil_keywords = [
        "soil", "fertilizer", "urea", "dap", "mop", "npk", "khad", "zinc", "potash", "nitrogen", "soil health", "soil test",
        "खाद", "उर्वरक", "यूरिया", "मिट्टी", "डीएपी", "पोटाश", "जिंक", "खत", "माती", "ਖਾਦ", "ਯੂਰੀਆ", "ਮਿੱਟੀ", "ਡੀਏਪੀ",
        "সার", "ইউরিয়া", "মাটি", "ఎరువు", "యూరియా", "మట్టి", "உரம்", "ખાતર", "ಗೊಬ್ಬರ", "വളം", "کھاد"
    ]
    if any(k in q for k in soil_keywords):
        return "soil"

    # 5. Climate Risk / Disaster Alert Intent
    risk_keywords = [
        "risk", "alert", "warning", "drought", "flood", "heat stress", "frost", "cold wave", "waterlogging",
        "जोखिम", "चेतावनी", "अलर्ट", "सूखा", "बाढ़", "पाला", "तुषार", "लू", "अतिवृष्टि", "ਧੋਕਾ", "ਖ਼ਤਰਾ", "ਚੇਤਾਵਨੀ", "ਸੋਕਾ"
    ]
    if any(k in q for k in risk_keywords):
        return "climate_risk"

    # 6. Crop Sowing / Variety / Cultivation Guide Intent
    crop_keywords = [
        "variety", "sowing", "seed", "crop", "beej", "plant", "yield", "season", "kharif", "rabi", "zaid",
        "किस्म", "बोने", "बीज", "पैदावार", "फसल", "रबी", "खरीफ", "ਕਿਸਮ", "ਬੀਜ", "ਫਸਲ", "রবী", "খরিফ",
        "విత్తనాలు", "పంట", "விதை", "பயிர்", "બિયારણ", "પાક", "ಬೆಳೆ", "വിള", "فصل"
    ]
    if any(k in q for k in crop_keywords):
        return "crop"

    return "general"


def run_agent(
    message: str,
    language: str = "English",
    location: str = "Delhi",
) -> Dict[str, Any]:
    """
    Run the KrishiRakshak agent and return a neat, well-structured response.
    Routes queries to targeted domain tools without cluttered data dumps.
    """
    llm = None
    try:
        llm = get_llm()
    except Exception as ex:
        logger.warning(f"LLM client setup notice: {ex}")

    # Resolve effective language and handle auto-detection
    lang_clean = (language or "auto").strip().lower()
    effective_language = "English"
    effective_code = "en"

    if lang_clean in ("auto", "auto-detect", "detect", "none", ""):
        effective_language, effective_code = detect_language(message)
    else:
        for name, code in SUPPORTED_LANGUAGES.items():
            if name.lower() in lang_clean or code == lang_clean:
                effective_language = name
                effective_code = code
                break
        else:
            effective_language, effective_code = detect_language(message)

    system_prompt = SYSTEM_PROMPTS.get(effective_language, SYSTEM_PROMPTS["English"])

    # 1. Identify primary intent
    intent = _classify_primary_intent(message)
    tool_outputs = []
    sources = []
    agent_steps = []

    # 2. Execute ONLY the single targeted domain tool based on primary intent
    if intent == "mandi":
        mandi_tool = get_market_tool()
        try:
            mandi_output = mandi_tool.func(f"{message} (Location: {location})")
            tool_outputs.append(mandi_output)
            sources.append("mandi_price_checker")
            agent_steps.append({"tool": "mandi_price_checker", "input": message, "output": mandi_output[:250]})
        except Exception as e:
            logger.warning(f"Mandi tool error: {e}")

    elif intent == "weather":
        wx_tool = get_weather_tool(location)
        try:
            wx_output = wx_tool.func(message)
            tool_outputs.append(wx_output)
            sources.append("weather_forecast")
            agent_steps.append({"tool": "weather_forecast", "input": message, "output": wx_output[:250]})
        except Exception as e:
            logger.warning(f"Weather tool error: {e}")

    elif intent == "pest_disease":
        pest_tool = get_pest_tool()
        try:
            p_output = pest_tool.func(message)
            tool_outputs.append(p_output)
            sources.append("pest_disease_control")
            agent_steps.append({"tool": "pest_disease_control", "input": message, "output": p_output[:250]})
        except Exception as e:
            logger.warning(f"Pest tool error: {e}")

    elif intent == "soil":
        soil_tool = get_soil_tool(location)
        try:
            s_output = soil_tool.func(message)
            tool_outputs.append(s_output)
            sources.append("soil_health_card_fertilizer")
            agent_steps.append({"tool": "soil_health_card_fertilizer", "input": message, "output": s_output[:250]})
        except Exception as e:
            logger.warning(f"Soil tool error: {e}")

    elif intent == "climate_risk":
        risk_tool = get_climate_risk_tool(location)
        try:
            r_output = risk_tool.func(message)
            tool_outputs.append(r_output)
            sources.append("climate_risk_early_warning")
            agent_steps.append({"tool": "climate_risk_early_warning", "input": message, "output": r_output[:250]})
        except Exception as e:
            logger.warning(f"Climate risk tool error: {e}")

    elif intent == "crop":
        crop_tool = get_crop_tool()
        try:
            c_output = crop_tool.func(message)
            tool_outputs.append(c_output)
            sources.append("crop_recommendation")
            agent_steps.append({"tool": "crop_recommendation", "input": message, "output": c_output[:250]})
        except Exception as e:
            logger.warning(f"Crop tool error: {e}")

    # If general or tool produced nothing, query RAG
    if not tool_outputs:
        try:
            rag_tool = _make_rag_tool()
            rag_output = rag_tool.func(message)
            if "No specific documentation found" not in rag_output and "failed" not in rag_output:
                tool_outputs.append(rag_output)
                sources.append("agricultural_knowledge_base")
                agent_steps.append({"tool": "agricultural_knowledge_base", "input": message, "output": rag_output[:250]})
        except Exception as e:
            logger.warning(f"RAG search error: {e}")

    # Build prompt for LLM synthesis
    context = "\n\n".join(tool_outputs) if tool_outputs else "No specific tool telemetry needed for this general query."
    augmented_message = f"{message} (Farmer's Location: {location})"

    lang_instructions = {
        "English": "Answer in clean, polite, structured English. Use 3 sections: 🎯 Direct Answer, 📌 Key Points (max 3 bullets), 💡 Farmer Tip. Do NOT dump unrelated crops, weather, or mandi prices.",
        "Hindi": "उत्तर पूरी तरह से सरल, शुद्ध और आदरणीय हिंदी में दें। ३ स्पष्ट भाग रखें: 🎯 मुख्य उत्तर, 📌 महत्वपूर्ण बिंदु (अधिकतम ३ बुलेट्स), 💡 किसान सलाह। अनावश्यक फसल या मौसम का डेटा बिल्कुल न दें।",
        "Punjabi": "ਸਰਲ ਅਤੇ ਸਪੱਸ਼ਟ ਪੰਜਾਬੀ (ਗੁਰਮੁਖੀ) ਵਿੱਚ ਜਵਾਬ ਦਿਓ। ੩ ਭਾਗ ਵਰਤੋ: 🎯 ਸਿੱਧਾ ਜਵਾਬ, 📌 ਮੁੱਖ ਨੁਕਤੇ, 💡 ਕਿਸਾਨ ਸਲਾਹ। ਕੋਈ ਵਾਧੂ ਜਾਣਕਾਰੀ ਨਾ ਦਿਓ।",
        "Bengali": "সহজ ও স্পষ্ট বাংলা ভাষায় উত্তর দিন। ৩টি অংশ রাখুন: 🎯 সরাসরি উত্তর, 📌 মূল পয়েন্ট, 💡 কৃষকের পরামর্শ।",
        "Telugu": "సరళమైన తెలుగులో సమాధానం ఇవ్వండి: 🎯 ప్రత్యక్ష సమాధానం, 📌 ముఖ్య విషయాలు, 💡 రైతు చిట్కా.",
        "Marathi": "सोप्या मराठीत उत्तर द्या: 🎯 मुख्य उत्तर, 📌 कृती मुद्दे, 💡 शेतकरी सल्ला.",
        "Tamil": "எளிய தமிழில் பதிலளிக்கவும்: 🎯 நேரடி பதில், 📌 முக்கிய புள்ளிகள், 💡 விவசாயி குறிப்பு.",
        "Gujarati": "સરળ ગુજરાતીમાં જવાબ આપો: 🎯 સીધો જવાબ, 📌 મુખ્ય મુદ્દાઓ, 💡 ખેડૂત સલાહ.",
        "Kannada": "ಸರಳ ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ: 🎯 ನೇರ ಉತ್ತರ, 📌 ಪ್ರಮುಖ ಅಂಶಗಳು, 💡 ರೈತ ಸಲಹೆ.",
        "Malayalam": "ലളിതമായ മലയാളത്തിൽ മറുപടി നൽകുക: 🎯 നേരിട്ടുള്ള മറുപടി, 📌 പ്രധാന കാര്യങ്ങൾ, 💡 കർഷക നിർദ്ദേശം.",
        "Urdu": "آسان اردو میں جواب دیں: 🎯 سیدھا جواب، 📌 اہم نکات، 💡 کسان مشورہ۔",
    }
    lang_instruction = lang_instructions.get(effective_language, lang_instructions["English"])

    full_prompt = (
        f"{system_prompt}\n\n"
        f"--- TARGETED FIELD DATA ---\n"
        f"{context}\n\n"
        f"--- FARMER QUERY ---\n"
        f"{augmented_message}\n\n"
        f"STRICT INSTRUCTION: {lang_instruction}\n"
        f"Keep the answer concise, neat, well-structured, and directly relevant to the farmer's question."
    )

    # 3. If LLM is available, synthesize the clean answer
    if llm is not None:
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=full_prompt),
            ])
            return {
                "response": response.content.strip(),
                "sources": sources,
                "agent_steps": agent_steps,
                "language": effective_language,
                "language_code": effective_code,
            }
        except Exception as e:
            logger.error(f"LLM synthesis error: {e}")

    # 4. Smart Clean Fallback (when no LLM key is configured)
    # Never dump raw telemetry; deliver structured, clean advice
    if tool_outputs:
        clean_data = tool_outputs[0].strip()

        fallback_headers = {
            "English": "🌾 **KrishiRakshak Agricultural Intelligence:**",
            "Hindi": "🌾 **कृषिरक्षक प्रामाणिक कृषि परामर्श:**",
            "Punjabi": "🌾 **ਕ੍ਰਿਸ਼ੀ ਰੱਖਿਅਕ ਪ੍ਰਮਾਣਿਤ ਖੇਤੀਬਾੜੀ ਸਲਾਹ:**",
            "Bengali": "🌾 **কৃষিরক্ষক প্রমাণিত কৃষি পরামর্শ:**",
            "Telugu": "🌾 **కృషిరక్షక్ ధృవీకరించిన వ్యవసాయ సలహా:**",
            "Marathi": "🌾 **कृषिरक्षक प्रमाणित शेती सल्ला:**",
            "Tamil": "🌾 **கிருஷிரக்ஷக் சரிபார்க்கப்பட்ட வேளாண்மை ஆலோசனை:**",
            "Gujarati": "🌾 **કૃષિરક્ષક પ્રમાણિત કૃષિ માર્ગદર્શન:**",
            "Kannada": "🌾 **ಕೃಷಿರಕ್ಷಕ್ ಪರಿಶೀಲಿಸಿದ ಕೃಷಿ ಸಲಹೆ:**",
            "Malayalam": "🌾 **കൃഷിരക്ഷക് കാർഷിക നിർദ്ദേശം:**",
            "Urdu": "🌾 **کرشی رکشاک مصدقہ زرعی ایڈوائزری:**",
        }
        header = fallback_headers.get(effective_language, fallback_headers["English"])

        return {
            "response": f"{header}\n\n{clean_data}",
            "sources": sources,
            "agent_steps": agent_steps,
            "language": effective_language,
            "language_code": effective_code,
        }

    # 5. Default guided response
    default_msg = {
        "English": (
            "🌾 **KrishiRakshak Advisory Assistant**\n\n"
            "Please ask a specific question such as:\n"
            "• *\"What is today's wheat mandi price in Delhi?\"*\n"
            "• *\"How to control Yellow Rust in wheat?\"*\n"
            "• *\"When should I apply the first urea split?\"*\n"
            "• *\"Will it rain in my region this week?\"*\n\n"
            "💡 *Tip: Enter your free Groq API key in the left sidebar for interactive conversational AI answers.*"
        ),
        "Hindi": (
            "🌾 **कृषिरक्षक कृषि परामर्श सहायक**\n\n"
            "कृपया अपना विशिष्ट सवाल पूछें, जैसे:\n"
            "• *\"दिल्ली मंडी में आज गेहूं का भाव क्या है?\"*\n"
            "• *\"गेहूं में पीली कुंगी (रस्ट) की रोकथाम कैसे करें?\"*\n"
            "• *\"गेहूं में यूरिया की पहली खुराक कब दें?\"*\n"
            "• *\"क्या इस हफ्ते मेरे क्षेत्र में बारिश होगी?\"*\n\n"
            "💡 *सुझाव: अधिक व्यक्तिगत और विस्तृत बातचीत के लिए बायीं ओर साइडबार में अपनी Groq API Key दर्ज करें।*"
        ),
        "Punjabi": (
            "🌾 **ਕ੍ਰਿਸ਼ੀ ਰੱਖਿਅਕ ਖੇਤੀਬਾੜੀ ਸਹਾਇਕ**\n\n"
            "ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਖਾਸ ਸਵਾਲ ਪੁੱਛੋ, ਜਿਵੇਂ:\n"
            "• *\"ਲੁਧਿਆਣਾ ਮੰਡੀ ਵਿੱਚ ਅੱਜ ਕਣਕ ਦਾ ਕੀ ਭਾਅ ਹੈ?\"*\n"
            "• *\"ਕਣਕ ਵਿੱਚ ਪੀਲੀ ਕੁੰਗੀ ਦਾ ਕੀ ਇਲਾਜ ਹੈ?\"*\n"
            "• *\"ਪਹਿਲੇ ਪਾਣੀ ਤੇ ਯੂਰੀਆ ਕਦੋਂ ਪਾਈਏ?\"*\n\n"
            "💡 *ਸੁਝਾਅ: ਵਿਸਤ੍ਰਿਤ AI ਗੱਲਬਾਤ ਲਈ ਖੱਬੇ ਪਾਸੇ ਸਾਈਡਬਾਰ ਵਿੱਚ ਆਪਣੀ Groq API Key ਦਰਜ ਕਰੋ।*"
        ),
    }

    return {
        "response": default_msg.get(effective_language, default_msg["English"]),
        "sources": sources,
        "agent_steps": agent_steps,
        "language": effective_language,
        "language_code": effective_code,
    }
