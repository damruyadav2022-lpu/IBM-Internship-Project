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
from agent.tools.soil_tool import get_soil_tool
from voice_assistant import detect_language, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


SYSTEM_PROMPTS = {
    "English": (
        "You are KrishiRakshak, an expert AI farming assistant for Indian farmers. "
        "You provide practical, accurate, and actionable agricultural advice. "
        "Respond in clear, simple English. Always mention the data sources you used."
    ),
    "Hindi": (
        "आप कृषि रक्षक हैं, भारतीय किसानों के लिए एक विशेषज्ञ AI सहायक। "
        "आप व्यावहारिक, सटीक और कार्रवाई योग्य कृषि सलाह देते हैं। "
        "हमेशा सरल ग्रामीण हिंदी भाषा में जवाब दें जो किसान समझ सकें। "
        "जब पर्याप्त जानकारी मिल जाए, तो व्यापक और उपयोगी उत्तर दें।"
    ),
    "Punjabi": (
        "ਤੁਸੀਂ ਕ੍ਰਿਸ਼ੀ ਰੱਖਿਅਕ (KrishiRakshak) ਹੋ, ਭਾਰਤੀ ਕਿਸਾਨਾਂ ਲਈ ਇੱਕ ਮਾਹਰ AI ਖੇਤੀਬਾੜੀ ਸਹਾਇਕ। "
        "ਤੁਸੀਂ ਵਿਹਾਰਕ, ਸਟੀਕ ਅਤੇ ਲਾਭਦਾਇਕ ਖੇਤੀ ਸਲਾਹ ਦਿੰਦੇ ਹੋ। "
        "ਹਮੇਸ਼ਾ ਸਰਲ, ਸਪੱਸ਼ਟ ਪੰਜਾਬੀ ਬੋਲੀ (ਗੁਰਮੁਖੀ ਲਿਪੀ) ਵਿੱਚ ਜਵਾਬ ਦਿਓ ਤਾਂ ਜੋ ਕਿਸਾਨ ਆਸਾਨੀ ਨਾਲ ਸਮਝ ਸਕਣ। "
        "ਜਦੋਂ ਬਿਮਾਰੀ ਜਾਂ ਕੀੜੇ ਬਾਰੇ ਪੁੱਛਿਆ ਜਾਵੇ, ਤਾਂ ਲੱਛਣ, ਕਾਰਨ, ਦੇਸੀ/ਜੈਵਿਕ ਇਲਾਜ ਅਤੇ ਰਸਾਇਣਕ ਰੋਕਥਾਮ ਸਪੱਸ਼ਟ ਦੱਸੋ।"
    ),
    "Bengali": (
        "আপনি কৃষিরক্ষক (KrishiRakshak), ভারতীয় কৃষকদের জন্য একজন বিশেষজ্ঞ এআই কৃষি সহকারী। "
        "সর্বদা স্পষ্ট, সহজ বাংলা ভাষায় কৃষকদের বোঝার উপযোগী পরামর্শ দিন। সার, কীটনাশক ও সেচ বিষয়ে সঠিক নির্দেশনা প্রদান করুন।"
    ),
    "Telugu": (
        "మీరు కృషిరక్షక్ (KrishiRakshak), భారతీయ రైతుల కోసం నిపుణులైన AI వ్యవసాయ సహాయకుడు. "
        "ఎల్లప్పుడూ స్పష్టమైన, సరళమైన తెలుగు భాషలో రైతులకు ఉపయోగకరమైన సలహాలు ఇవ్వండి. ఎరువులు, తెగుళ్ల నివారణ గురించి పూర్తి వివరాలు అందించండి."
    ),
    "Marathi": (
        "तुम्ही कृषिरक्षक (KrishiRakshak) आहात, भारतीय शेतकऱ्यांसाठी एक तज्ज्ञ AI कृषी सल्लागार. "
        "शेतकऱ्यांना समजेल अशा सोप्या मराठी भाषेत अचूक आणि व्यावहारिक शेतीविषयक मार्गदर्शन करा."
    ),
    "Tamil": (
        "நீங்கள் கிருஷிரக்ஷக் (KrishiRakshak), இந்திய விவசாயிகளுக்கான ஒரு சிறந்த AI வேளாண்மை உதவியாளர். "
        "விவசாயிகளுக்கு எளிய, தெளிவான தமிழ் மொழியில் பயிர் பாதுகாப்பு, உரம் மற்றும் பாசன ஆலோசனைகளை வழங்குங்கள்."
    ),
    "Gujarati": (
        "તમે કૃષિરક્ષક (KrishiRakshak) છો, ભારતીય ખેડૂતો માટે એક નિષ્ણાત AI ખેતી સહાયક. "
        "હંમેશા સરળ ગુજરાતી ભાષામાં ખેડૂતો માટે વ્યવહારુ અને સચોટ કૃષિ સલાહ આપો."
    ),
    "Kannada": (
        "ನೀವು ಕೃಷಿರಕ್ಷಕ್ (KrishiRakshak), ಭಾರತೀಯ ರೈತರಿಗಾಗಿ ಪರಿಣಿತ AI ಕೃಷಿ ಸಹಾಯಕ. "
        "ರೈತರಿಗೆ ಸುಲಭವಾಗಿ ಅರ್ಥವಾಗುವ ಸರಳ ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ನಿಖರವಾದ ಕೃಷಿ ಸಲಹೆಗಳನ್ನು ನೀಡಿ."
    ),
    "Malayalam": (
        "നിങ്ങൾ കൃഷിരക്ഷക് (KrishiRakshak) ആണ്, ഇന്ത്യൻ കർഷകർക്കായുള്ള വിദഗ്ദ്ധ AI കാർഷിക സഹായി. "
        "ലളിതമായ മലയാളത്തിൽ കർഷകർക്ക് പ്രയോജനപ്പെടുന്ന കൃഷി ഉപദേശങ്ങളും കീടനിയന്ത്രണ മാർഗ്ഗങ്ങളും നൽകുക."
    ),
    "Urdu": (
        "آپ کرشی رکشاک (KrishiRakshak) ہیں، ہندوستانی کسانوں کے لیے ماہر AI زرعی معاون۔ "
        "ہمیشہ صاف، آسان اردو زبان میں فصلوں، کھاد اور کیڑوں کے تدارک کے بارے میں مفید رہنمائی فراہم کریں۔"
    ),
}


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

    msg_lower = message.lower()

    # Domain tools
    tool_instances = {
        "weather_forecast": get_weather_tool(location),
        "crop_recommendation": get_crop_tool(),
        "pest_disease_control": get_pest_tool(),
        "disease_info": create_disease_tool(),
        "mandi_price_checker": get_market_tool(),
        "climate_risk_early_warning": get_climate_risk_tool(location),
        "soil_health_card_fertilizer": get_soil_tool(location),
    }

    keyword_map = {
        "soil_health_card_fertilizer": [
            "soil", "fertilizer", "urea", "dap", "mop", "npk", "soil health", "soil test",
            "khad", "urvarak", "zinc", "potash", "phosphorus", "nitrogen", "pau", "ludhiana",
            # Hindi / Marathi
            "खाद", "उर्वरक", "यूरिया", "मिट्टी", "सॉइल हेल्थ", "डीएपी", "पोटाश", "जिंक", "खत", "माती", "जमीन",
            # Punjabi
            "ਖਾਦ", "ਯੂਰੀਆ", "ਮਿੱਟੀ", "ਸੋਇਲ ਹੈਲਥ", "ਡੀਏਪੀ", "ਪੋਟਾਸ਼", "ਜ਼ਿੰਕ",
            # Bengali
            "সার", "ইউরিয়া", "মাটি", "ডিএপি", "পটাশ",
            # Telugu
            "ఎరువు", "యూరియా", "మట్టి", "పొటాష్",
            # Tamil
            "உரம்", "யூரியா", "மண்",
            # Gujarati
            "ખાતર", "જમીન", "માટી", "યુરિયા",
            # Kannada
            "ಗೊಬ್ಬರ", "ಯೂರಿಯಾ", "ಮಣ್ಣು",
            # Malayalam
            "വളം", "യൂറിയ", "മണ്ണ്",
            # Urdu
            "کھاد", "مٹی", "یوریا"
        ],
        "climate_risk_early_warning": [
            "risk", "alert", "warning", "drought", "flood", "heat stress", "frost", "cold wave",
            "heavy rain", "waterlogging", "soil moisture", "lodging", "climate risk", "early warning",
            "sukha", "badh", "pala", "chilling",
            # Hindi / Marathi
            "जोखिम", "चेतावनी", "अलर्ट", "सूखा", "बाढ़", "पाला", "तुषार", "लू", "अतिवृष्टि", "ओलावृष्टि", "नमी", "धोका", "दुष्काळ",
            # Punjabi
            "ਖ਼ਤਰਾ", "ਚੇਤਾਵਨੀ", "ਸੋਕਾ", "ਹੜ੍ਹ", "ਪਾਲਾ", "ਤਪਸ਼", "ਲੂ", "ਸਿੱਲ੍ਹ",
            # Bengali
            "ঝুঁকি", "সতর্কতা", "খরা", "বন্যা", "অতিবৃষ্টি",
            # Telugu
            "ప్రమాదం", "హెచ్చరిక", "కరువు", "వరదలు",
            # Tamil
            "எச்சரிக்கை", "வறட்சி", "வெள்ளம்",
            # Gujarati
            "ચેતવણી", "જોખમ", "દુષ્કાળ", "પૂર",
            # Kannada
            "ಎಚ್ಚರಿಕೆ", "ಬರಗಾಲ", "ಪ್ರವಾಹ",
            # Malayalam
            "മുന്നറിയിപ്പ്", "വരൾച്ച", "വെള്ളപ്പൊക്കം",
            # Urdu
            "انتباہ", "خطرہ", "خشک سالی", "سیलाब"
        ],
        "weather_forecast": [
            "weather", "rain", "temperature", "forecast", "mausam", "barish",
            "humidity", "wind", "storm", "climate", "hawa", "thand", "garmi", "advisory",
            # Hindi / Marathi
            "मौसम", "बारिश", "बरसात", "तापमान", "हवा", "ठंड", "गर्मी", "सलाह", "हवामान", "पाऊस", "गारपीट",
            # Punjabi
            "ਮੌਸਮ", "ਮੀਂਹ", "ਤਾਪਮਾਨ", "ਹਵਾ", "ਬਾਰਿਸ਼", "ਸਲਾਹ",
            # Bengali
            "আবহাওয়া", "বৃষ্টি", "তাপমাত্রা", "ঝড়",
            # Telugu
            "వాతావరణం", "వర్షం", "ఉష్ణోగ్రత",
            # Tamil
            "வானிலை", "மழை", "வெப்பநிலை",
            # Gujarati
            "હવામાન", "વરસાદ", "તાપમાન",
            # Kannada
            "ಹವಾಮಾನ", "ಮಳೆ", "ತಾಪಮಾನ",
            # Malayalam
            "കാലാവസ്ഥ", "മഴ", "താപനില",
            # Urdu
            "موسم", "بارش", "درجہ حرارت"
        ],
        "crop_recommendation": [
            "crop", "season", "kharif", "rabi", "zaid", "sow", "plant", "fasal",
            "beej", "variety", "yield", "gehun", "wheat", "rice", "chawal", "dhan",
            "makka", "corn", "cotton", "kapas", "mustard", "sarson", "potato", "aloo",
            "sowing", "harvest",
            # Hindi / Marathi
            "फसल", "खरीफ", "रबी", "जायद", "बोने", "बीज", "पैदावार", "किस्म", "गेहूं",
            "धान", "चावल", "मक्का", "कपास", "सरसों", "आलू", "पीक", "बियाणे", "उत्पादन",
            # Punjabi
            "ਫਸਲ", "ਕਣਕ", "ਝੋਨਾ", "ਬੀਜ", "ਖਰੀਫ", "ਰਬੀ", "ਮੱਕੀ", "ਕਪਾਹ", "ਆਲੂ",
            # Bengali
            "ফসল", "ধান", "গম", "বীজ", "রবি", "খরিফ",
            # Telugu
            "పంట", "వరి", "గోధుమ", "విత్తనాలు",
            # Tamil
            "பயிர்", "நெல்", "கோதுமை", "விதை",
            # Gujarati
            "પાક", "ઘઉં", "ડાંગર", "બિયારણ",
            # Kannada
            "ಬೆಳೆ", "ಭತ್ತ", "ಗೋಧಿ", "ಬೀಜ",
            # Malayalam
            "വിള", "നെല്ല്", "ഗോതമ്പ്", "വിത്ത്",
            # Urdu
            "فصل", "گندم", "چاول", "بیج"
        ],
        "pest_disease_control": [
            "pest", "insect", "keet", "keda", "illia", "sundhi", "spray",
            "pesticide", "neem", "caterpillar", "aphid", "borer",
            # Hindi / Marathi
            "कीट", "कीड़ा", "रोकथाम", "दवा", "कीटनाशक", "नीम", "स्प्रे", "कीड", "अळी", "फवारणी",
            # Punjabi
            "ਕੀੜੇ", "ਸੁੰਡੀ", "ਸਪਰੇਅ", "ਕੀਟਨਾਸ਼ਕ", "ਰੋਕਥਾਮ", "ਤੇਲਾ",
            # Bengali
            "কীটপতঙ্গ", "পোকা", "কীটনাশক", "স্প্রে",
            # Telugu
            "కీటకాలు", "పురుగు", "పురుగుమందు", "స్ప్రే",
            # Tamil
            "பூச்சி", "பூச்சிக்கொல்லி", "தெளிப்பு",
            # Gujarati
            "જીવાત", "ઇયળ", "જંતુનાશક", "છંટકાવ",
            # Kannada
            "ಕೀಟ", "ಹುಳು", "ಕೀಟನಾಶಕ",
            # Malayalam
            "കീടങ്ങൾ", "കീടനാശിനി", "സ്പ്രേ",
            # Urdu
            "کیڑے", "کیڑے مار دوا", "سپرے"
        ],
        "disease_info": [
            "disease", "leaf", "symptom", "treatment", "bimari", "rog", "blight",
            "rust", "spot", "yellow", "curl", "fungus", "wilt", "rot",
            # Hindi / Marathi
            "रोग", "बीमारी", "झुलसा", "जंग", "धब्बा", "पत्ती", "फफूंद", "बुरशी", "रोगराई",
            # Punjabi
            "ਬਿਮਾਰੀ", "ਪੱਤੇ", "ਪੱਤਿਆਂ", "ਪੀਲੇ", "ਧੱਬੇ", "ਕੁੰਗੀ", "ਝੁਲਸਾ", "ਫੰਗਸ", "ਰੋਗ", "ਟਮਾਟਰ",
            # Bengali
            "রোগ", "পাতা", "দাগ", "হলুদ", "ছত্রাক",
            # Telugu
            "తెగులు", "వ్యాధి", "ఆకులు", "మచ్చలు", "శిలీంధ్రం",
            # Tamil
            "நோய்", "இலை", "புள்ளி", "பூஞ்சை",
            # Gujarati
            "રોગ", "પાંદડા", "ડાઘ", "ફૂગ",
            # Kannada
            "ರೋಗ", "ಎಲೆ", "ಮಚ್ಚೆ", "ಶಿಲೀಂಧ್ರ",
            # Malayalam
            "രോഗം", "ഇല", "പുള്ളി", "ഫംഗസ്",
            # Urdu
            "بیماری", "پتے", "دھبے", "پھپھوندی"
        ],
        "mandi_price_checker": [
            "price", "mandi", "rate", "bhav", "daam", "market", "sell", "cost",
            "msp", "quintal", "rupee", "rs", "bazar", "kharid",
            # Hindi / Marathi
            "मंडी", "भाव", "दाम", "रेट", "मूल्य", "बाजार", "टमाटर", "क्विंटल", "खरीद", "बेचने", "दर", "बाजारभाव",
            # Punjabi
            "ਮੰਡੀ", "ਭਾਅ", "ਰੇਟ", "ਕੀਮਤ", "ਕੁਇੰਟਲ",
            # Bengali
            "মন্ডি", "দাম", "দর", "বাজার", "মূল্য", "টমেটো",
            # Telugu
            "ధర", "మార్కెట్", "రేటు", "మండి", "టమాటా",
            # Tamil
            "விலை", "சந்தை", "மண்டி", "தக்காளி",
            # Gujarati
            "ભાવ", "બજાર", "મંડી", "કિંમત", "ટમેટા",
            # Kannada
            "ಬೆಲೆ", "ಮಾರುಕಟ್ಟೆ", "ದರ", "ಟೊಮೆಟೊ",
            # Malayalam
            "വില", "വിപണി", "മാർക്കറ്റ്", "തക്കാളി",
            # Urdu
            "بھاؤ", "منڈی", "قیمت", "ریٹ", "بازار"
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

    lang_instructions = {
        "English": "Answer in clear, polite, structured English with practical recommendations and an action checklist.",
        "Hindi": "कृपया उत्तर पूरी तरह से शुद्ध, सरल और आदरणीय हिंदी (Devanagari script) में दें। व्यावहारिक सुझाव, रोकथाम के उपाय और स्पष्ट बुलेट पॉइंट्स या तालिका का उपयोग करें।",
        "Punjabi": "ਕਿਰਪਾ ਕਰਕੇ ਪੂਰਾ ਜਵਾਬ ਸਰਲ ਅਤੇ ਸਪੱਸ਼ਟ ਪੰਜਾਬੀ (Gurmukhi script) ਵਿੱਚ ਦਿਓ। ਕਿਸਾਨ ਵੀਰਾਂ ਲਈ ਵਿਹਾਰਕ ਸਲਾਹ ਅਤੇ ਬੁਲੇਟ ਪੁਆਇੰਟਾਂ ਦੀ ਵਰਤੋਂ ਕਰੋ।",
        "Bengali": "দয়া করে সহজ, স্পষ্ট ও ব্যবহারিক বাংলা ভাষায় উত্তর দিন। বুলেট পয়েন্ট ও সারসংক্ষেপের মাধ্যমে কৃষকের বোধগম্য নির্দেশিকা দিন।",
        "Telugu": "దయచేసి స్పష్టమైన, సరళమైన తెలుగు భాషలో సమాధానం ఇవ్వండి. రైతులకు ఉపయోగపడే కార్యాచరణ అంశాలు మరియు బుల్లెట్ పాయింట్లను ఉపయోగించండి.",
        "Marathi": "कृपया सोप्या, अचूक आणि व्यावहारिक मराठी भाषेत उत्तर द्या. शेतकऱ्यांसाठी आवश्यक सूचना व मुद्द्यांचा समावेश करा.",
        "Tamil": "தயவுசெய்து எளிய, தெளிவான தமிழ் மொழியில் பதிலளிக்கவும். விவசாயிகளுக்கான நடைமுறை ஆலோசனைகளை புல்லட் புள்ளிகளில் வழங்கவும்.",
        "Gujarati": "કૃપા કરીને સરળ, સ્પષ્ટ ગુજરાતી ભાષામાં જવાબ આપો. ખેડૂતો માટે વ્યવહારુ સલાહ અને મુદ્દાઓ જણાવો.",
        "Kannada": "ದಯವಿಟ್ಟು ಸರಳ, ಸ್ಪಷ್ಟ ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ಉತ್ತರಿಸಿ. ರೈತರಿಗೆ ಅನುಕೂಲವಾಗುವ ಪ್ರಾಯೋಗಿಕ ಸಲಹೆಗಳನ್ನು ಬುಲೆಟ್ ಪಾಯಿಂಟ್‌ಗಳಲ್ಲಿ ನೀಡಿ.",
        "Malayalam": "ദയവായി ലളിതമായ മലയാളത്തിൽ കർഷകർക്ക് പ്രയോജനപ്പെടുന്ന പ്രായോഗിക നിർദ്ദേശങ്ങൾ നൽകുക.",
        "Urdu": "براہ کرم آسان، واضح اردو زبان میں جواب دیں۔ کسانوں کے لیے عملی تجاویز اور اقدامات بلٹ پوائنٹس میں پیش کریں۔",
    }
    lang_instruction = lang_instructions.get(effective_language, lang_instructions["English"])

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
                "language": effective_language,
                "language_code": effective_code,
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

        fallback_headers = {
            "English": "🌾 **Verified Agricultural Advisory & Real-Time Intelligence:**",
            "Hindi": "🌾 **सत्यापित कृषि परामर्श एवं मौसम बुलेटिन (रियल-टाइम डेटा):**",
            "Punjabi": "🌾 **ਪ੍ਰਮਾਣਿਤ ਖੇਤੀਬਾੜੀ ਸਲਾਹ ਅਤੇ ਮੌਸਮ ਜਾਣਕਾਰੀ (ਰੀਅਲ-ਟਾਈਮ ਡੇਟਾ):**",
            "Bengali": "🌾 **যাচাইকৃত কৃষি পরামর্শ ও আবহাওয়া বুলেটিন (রিয়েল-টাইম তথ্য):**",
            "Telugu": "🌾 **ధృవీకరించబడిన వ్యవసాయ సలహాలు మరియు వాతావరణ బులెటిన్ (రియల్-టైమ్ డేటా):**",
            "Marathi": "🌾 **सत्यापित कृषी सल्ला आणि हवामान अंदाज (रिअल-टाइम डेटा):**",
            "Tamil": "🌾 **சரிபார்க்கப்பட்ட வேளாண்மை ஆலோசனை மற்றும் வானிலை அறிக்கை (நேரலை தரவு):**",
            "Gujarati": "🌾 **ચકાસાયેલ કૃષિ સલાહ અને હવામાન બુલેટિન (રીઅલ-ટાઇમ ડેટા):**",
            "Kannada": "🌾 **ಪರಿಶೀಲಿಸಿದ ಕೃಷಿ ಸಲಹೆ ಮತ್ತು ಹವಾಮಾನ ಮಾಹಿತಿ (ನೈಜ-ಸಮಯದ ಡೇಟಾ):**",
            "Malayalam": "🌾 **സ്ഥിരീകരിച്ച കാർഷിക നിർദ്ദേശങ്ങളും കാലാവസ്ഥാ ബുള്ളറ്റിനും:**",
            "Urdu": "🌾 **تصدیق شدہ زرعی ایڈوائزری اور موسم کا بلیٹن (ریئل ٹائم ڈیٹا):**",
        }
        fallback_footers = {
            "English": "💡 *Tip: For interactive conversational answers and custom follow-ups, activate your Groq API key in the left sidebar or in Streamlit Cloud Secrets.*",
            "Hindi": "💡 *सुझाव: अधिक विस्तृत AI बातचीत और व्यक्तिगत सवालों के लिए बायीं ओर साइडबार में अपनी Groq API Key दर्ज करें।*",
            "Punjabi": "💡 *ਸੁਝਾਅ: ਵਿਸਤ੍ਰਿਤ AI ਜਵਾਬਾਂ ਲਈ ਖੱਬੇ ਪਾਸੇ ਸਾਈਡਬਾਰ ਵਿੱਚ ਆਪਣੀ Groq API Key ਦਰਜ ਕਰੋ।*",
            "Bengali": "💡 *পরামর্শ: বিস্তারিত AI কথোপকথনের জন্য বাম সাইডবারে আপনার Groq API Key দিন।*",
            "Telugu": "💡 *సూచన: పూర్తి AI సలహాల కోసం ఎడమ సైడ్‌బార్‌లో మీ Groq API Key నమోదు చేయండి.*",
            "Marathi": "💡 *टीप: अधिक तपशीलवार AI उत्तरांसाठी डाव्या साइडबारमध्ये तुमची Groq API Key प्रविष्ट करा.*",
            "Tamil": "💡 *குறிப்பு: விரிவான AI பதில்களுக்கு இடது பக்கப்பட்டியில் உங்கள் Groq API Key உள்ளிடவும்.*",
            "Gujarati": "💡 *સલાહ: વિગતવાર AI વાતચીત માટે ડાબી સાઇડબારમાં તમારી Groq API Key દાખલ કરો.*",
            "Kannada": "💡 *ಸಲಹೆ: ವಿವರವಾದ AI ಉತ್ತರಗಳಿಗಾಗಿ ಎಡ ಸೈಡ್‌ಬಾರ್‌ನಲ್ಲಿ ನಿಮ್ಮ Groq API Key ನಮೂದಿಸಿ.*",
            "Malayalam": "💡 *നിർദ്ദേശം: കൂടുതൽ AI മറുപടികൾക്കായി ഇടതു സൈഡ്ബാറിൽ Groq API Key ചേർക്കുക.*",
            "Urdu": "💡 *مشورہ: تفصیلی AI گفتگو کے لیے بائیں سائڈبار میں اپنی Groq API Key درج کریں۔*",
        }

        header = fallback_headers.get(effective_language, fallback_headers["English"])
        footer_tip = fallback_footers.get(effective_language, fallback_footers["English"])

        return {
            "response": f"{header}\n\n{retrieved_data}\n\n---\n{footer_tip}",
            "sources": sources,
            "agent_steps": agent_steps,
            "language": effective_language,
            "language_code": effective_code,
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
        "language": effective_language,
        "language_code": effective_code,
    }
