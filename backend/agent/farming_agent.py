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
try:
    from agent.rag_pipeline import similarity_search
except Exception as e:
    similarity_search = None
    logger = logging.getLogger(__name__)
    logger.warning(f"ChromaDB RAG pipeline disabled or uninitialized: {e}")
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
        "You are KrishiRakshak AI, an expert agricultural advisor and scientist for Indian farmers.\n"
        "Provide in-depth, comprehensive, ChatGPT-style agronomic guidance that is crystal clear, actionable, and verified by PAU / ICAR standards.\n\n"
        "GUIDELINES FOR IN-DEPTH EXPERT ADVICE:\n"
        "1. Focus deeply on the farmer's specific question (crop, pest, disease, fertilizer, irrigation, or market).\n"
        "2. Provide complete technical depth without unnecessary brevity. Structure your response into these well-defined sections:\n"
        "   🎯 **Direct Diagnosis & Overview**: Detailed explanation of the condition, growth stage, or market reality.\n"
        "   🌿 **Organic & Biological Control (IPM)**: Natural methods, neem formulations, beneficial bio-agents, and preparation techniques.\n"
        "   💊 **Recommended Chemical Control & Exact Dosages**: Specific CIBRC-approved active ingredients, exact dosage per acre and per 15-litre knapsack sprayer pump, and water volume.\n"
        "   🚜 **Agronomic & Cultural Practices**: Irrigation schedules (e.g. CRI stage), fertilizer top-dressing ratios, soil management, and preventive spacing.\n"
        "   💡 **Scientist Pro-Tips**: Critical field precautions, optimal spraying hours (dew/wind), safety gear, and waiting period before harvest."
    ),
    "Hindi": (
        "आप कृषि रक्षक (KrishiRakshak AI) हैं — भारतीय किसानों के लिए एक उच्च-स्तरीय कृषि वैज्ञानिक एवं सलाहकार।\n"
        "किसान के हर प्रश्न का ChatGPT जैसा विस्तृत, गहन, स्पष्ट और वैज्ञानिक रूप से सत्यापित (PAU/ICAR मानक) उत्तर दें।\n\n"
        "विस्तृत उत्तर संरचना के नियम:\n"
        "1. किसान ने जिस फसल, कीट, रोग, सिंचाई या खाद के बारे में पूछा है, उस पर गहन और पूर्ण जानकारी दें। संक्षिप्त या अधूरा उत्तर न दें।\n"
        "2. उत्तर को निम्नलिखित ५ स्पष्ट व सुंदर शीर्षकों में प्रस्तुत करें:\n"
        "   🎯 **१. स्थिति की पहचान एवं वैज्ञानिक विश्लेषण**: समस्या का कारण, लक्षण और फसल पर प्रभाव।\n"
        "   🌿 **२. जैविक व देसी रोकथाम (IPM)**: नीम तेल, ट्राइकोडर्मा, फेरोमोन ट्रैप या जैव-नियंत्रक का सटीक प्रयोग।\n"
        "   💊 **३. अनुशंसित रासायनिक उपचार एवं सटीक खुराक**: अनुमोदित कीटनाशक/फफूंदनाशक, प्रति एकड़ व प्रति 15 लीटर पंप की सटीक मात्रा (एमएल/ग्राम) और पानी की मात्रा।\n"
        "   🚜 **४. सस्य क्रियाएं, खाद व सिंचाई प्रबंधन**: बुवाई, निराई-गुड़ाई, सिंचाई की सही अवस्था (जैसे CRI), और यूरिया/पोटाश की सही खुराक।\n"
        "   💡 **५. कृषि वैज्ञानिक विशेष सलाह व सावधानियां**: छिड़काव का सही समय (सुबह/शाम), हवा का रुख, सुरक्षा और फसल कटाई से पूर्व प्रतीक्षा समय।"
    ),
    "Punjabi": (
        "ਤੁਸੀਂ ਕ੍ਰਿਸ਼ੀ ਰੱਖਿਅਕ (KrishiRakshak AI) ਹੋ — ਭਾਰਤੀ ਕਿਸਾਨਾਂ ਲਈ ਇੱਕ ਉੱਚ-ਪੱਧਰੀ ਖੇਤੀਬਾੜੀ ਵਿਗਿਆਨੀ ਅਤੇ ਮਾਹਰ ਸਲਾਹਕਾਰ।\n"
        "ਕਿਸਾਨ ਦੇ ਹਰ ਸਵਾਲ ਦਾ ਪੂਰਾ ਵਿਸਤ੍ਰਿਤ, ਡੂੰਘਾ ਅਤੇ ਪੀਏਯੂ ਲੁਧਿਆਣਾ (PAU) ਪ੍ਰਮਾਣਿਤ ਜਵਾਬ ਦਿਓ।\n\n"
        "ਜਵਾਬ ਦੀ ਬਣਤਰ:\n"
        "1. 🎯 **ਸਮੱਸਿਆ ਦੀ ਪਛਾਣ ਅਤੇ ਪ੍ਰਭਾਵ**: ਬਿਮਾਰੀ ਜਾਂ ਕੀੜੇ ਦੇ ਮੁੱਖ ਲੱਛਣ।\n"
        "2. 🌿 **ਜੈਵਿਕ ਅਤੇ ਦੇਸੀ ਰੋਕਥਾਮ**: ਨਿੰਮ ਦਾ ਅਰਕ ਜਾਂ ਜੈਵਿਕ ਤਰੀਕੇ।\n"
        "3. 💊 **ਸਿਫਾਰਸ਼ ਕੀਤੀਆਂ ਰਸਾਇਣਕ ਦਵਾਈਆਂ ਅਤੇ ਸਹੀ ਖੁਰਾਕ**: ਪ੍ਰਤੀ ਏਕੜ ਅਤੇ ਪ੍ਰਤੀ ਪੰਪ ਸਹੀ ਮਾਤਰਾ।\n"
        "4. 🚜 **ਖਾਦ ਅਤੇ ਸਿੰਚਾਈ ਪ੍ਰਬੰਧਨ**: ਪਹਿਲੇ ਪਾਣੀ (CRI) ਅਤੇ ਖਾਦ ਪਾਉਣ ਦਾ ਸਹੀ ਸਮਾਂ।\n"
        "5. 💡 **ਵਿਗਿਆਨਕ ਸਲਾਹ**: ਛਿੜਕਾਅ ਦਾ ਸਹੀ ਸਮਾਂ ਅਤੇ ਸਾਵਧਾਨੀਆਂ।"
    ),
    "Bengali": (
        "আপনি কৃষিরক্ষক (KrishiRakshak AI) — ভারতীয় কৃষকদের জন্য একজন প্রধান কৃষি বিজ্ঞানী ও বিশেষজ্ঞ।\n"
        "কৃষকের প্রশ্নের চ্যাটজিপিটি-র মতো বিশদ, গভীর, সুনির্দিষ্ট এবং বৈজ্ঞানিক সমাধান দিন: 🎯 পরিস্থিতি বিশ্লেষণ, 🌿 জৈব ও আইপিএম ব্যবস্থাপনা, 💊 অনুমোদিত রাসায়নিক ও সঠিক মাত্রা (প্রতি একর ও প্রতি স্প্রেয়ার), 🚜 সেচ ও সার ব্যবস্থাপনা, 💡 বিশেষজ্ঞ কৃষি পরামর্শ।"
    ),
    "Telugu": (
        "మీరు కృషిరక్షక్ (KrishiRakshak AI) — రైతులకు సమగ్రమైన మరియు శాస్త్రీయమైన వ్యవసాయ సలహాదారు.\n"
        "పూర్తి వివరాలతో కూడిన లోతైన సమాధానం ఇవ్వండి: 🎯 సమస్య నిర్ధారణ, 🌿 సేంద్రీయ & జీవ నియంత్రణ, 💊 రసాయన మందులు & ఖచ్చితమైన మోతాదులు (ఎకరాకు/పంప్ కు), 🚜 నీటి పారుదల & ఎరువుల యాజమాన్యం, 💡 శాస్త్రవేత్తల ప్రత్యేక సూచనలు."
    ),
    "Marathi": (
        "तुम्ही कृषिरक्षक (KrishiRakshak AI) आहात — भारतीय शेतकऱ्यांसाठी उच्च दर्जाचे कृषी शास्त्रज्ञ व सल्लागार.\n"
        "सविस्तर आणि सखोल तांत्रिक मार्गदर्शन करा: 🎯 रोगाची/किडीची लक्षणे व कारणे, 🌿 जैविक व सेंद्रिय नियंत्रण (IPM), 💊 शिफारस केलेली रसायने व अचूक प्रमाण (प्रति एकर व १५ लिटर पंपासाठी), 🚜 खत व पाणी व्यवस्थापन, 💡 कृषी तज्ज्ञ महत्त्वाच्या टिप्स."
    ),
    "Tamil": (
        "நீங்கள் கிருஷிரக்ஷக் (KrishiRakshak AI) — விவசாயிகளுக்கான நிபுணத்துவ வேளாண்மை விஞ்ஞானி.\n"
        "ஆழமான, விரிவான மற்றும் பயனுள்ள தகவல்களை வழங்கவும்: 🎯 பிரச்சனை அடையாளம், 🌿 இயற்கை மற்றும் உயிரியல் கட்டுப்பாடு, 💊 பரிந்துரைக்கப்பட்ட ரசாயனங்கள் மற்றும் சரியான அளவுகள், 🚜 உர மற்றும் பாசன மேலாண்மை, 💡 வேளாண் விஞ்ஞானி முக்கிய ஆலோசனைகள்."
    ),
    "Gujarati": (
        "તમે કૃષિરક્ષક (KrishiRakshak AI) છો — ભારતીય ખેડૂતો માટે તજજ્ઞ કૃષિ વૈજ્ઞાનિક.\n"
        "સંપૂર્ણ વિગતવાર અને વૈજ્ઞાનિક માર્ગદર્શન આપો: 🎯 સ્થિતિનું વિશ્લેષણ, 🌿 જૈવિક નિયંત્રણ, 💊 રાસાયણિક ઉપચાર અને ચોક્કસ ડોઝ (એકર/પંપ દીઠ), 🚜 ખાતર અને પિયત વ્યવસ્થાપન, 💡 કૃષિ નિષ્ણાત સલાહ."
    ),
    "Kannada": (
        "ನೀವು ಕೃಷಿರಕ್ಷಕ್ (KrishiRakshak AI) — ರೈತರಿಗೆ ತಜ್ಞ ಕೃಷಿ ವಿಜ್ಞಾನಿ ಮತ್ತು ಸಲಹೆಗಾರ.\n"
        "ಸಂಪೂರ್ಣ ವಿವರವಾದ ಮತ್ತು ಕ್ರಿಯಾತ್ಮಕ ಮಾರ್ಗದರ್ಶನ ನೀಡಿ: 🎯 ರೋಗ ಲಕ್ಷಣಗಳ ವಿಶ್ಲೇಷಣೆ, 🌿 ಜೈವಿಕ ನಿಯಂತ್ರಣ, 💊 ಶಿಫಾರಸು ಮಾಡಿದ ರಾಸಾಯನಿಕಗಳು ಮತ್ತು ನಿಖರ ಡೋಸ್, 🚜 ಗೊಬ್ಬರ ಮತ್ತು ನೀರಾವರಿ ನಿರ್ವಹಣೆ, 💡 ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆಗಳು."
    ),
    "Malayalam": (
        "നിങ്ങൾ കൃഷിരക്ഷക് (KrishiRakshak AI) ആണ് — കർഷകർക്കുള്ള വിദഗ്ദ്ധ കാർഷിക ശാസ്ത്രജ്ഞൻ.\n"
        "വിശദവും സമഗ്രവുമായ മാർഗ്ഗനിർദ്ദേശം നൽകുക: 🎯 രോഗ നിർണ്ണയം, 🌿 ജൈവ നിയന്ത്രണം, 💊 രാസ മരുന്നുകളും കൃത്യമായ അളവും (ഏക്കറിന്/സ്പ്രേയറിന്), 🚜 വളം, ജലസേചന ക്രമം, 💡 ശാസ്ത്രജ്ഞന്റെ പ്രധാന നിർദ്ദേശങ്ങൾ."
    ),
    "Urdu": (
        "آپ کرشی رکشاک (KrishiRakshak AI) ہیں — کسانوں کے لیے ایک مستند زرعی سائنسدان اور مشیر۔\n"
        "مکمل، گہری اور جامع رہنمائی فراہم کریں: 🎯 مرض یا مسئلے کی تشخیص، 🌿 قدرتی و حیاتیاتی تدارک، 💊 کیمیائی ادویات اور درست مقدار (فی ایکڑ اور فی پمپ)، 🚜 کھاد اور آبپاشی کا شیڈول، 💡 زرعی ماہرین کے مفید مشورے۔"
    ),
}


def _make_rag_tool() -> Tool:
    """Create a RAG knowledge base search tool."""
    def rag_search(query: str) -> str:
        if similarity_search is None:
            return "Knowledge base search temporarily unavailable."
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

    # 3. Pest / Disease / Spray / Cure Intent (High Priority)
    pest_keywords = [
        "pest", "insect", "disease", "spray", "pesticide", "fungus", "blight", "rust", "curl", "bollworm", "aphid",
        "whitefly", "termite", "keet", "keda", "bimari", "rog", "leaf curl", "wilt", "rot", "borer",
        "कीट", "कीड़ा", "रोग", "बीमारी", "झुलसा", "जंग", "धब्बा", "दवा", "दवाई", "कीटनाशक", "स्प्रे", "फफूंद", "कीड", "अळी",
        "कुंगी", "रस्ट", "रतुआ", "पीला रतुआ", "पीली कुंगी", "सुंडी", "गुलाबी सुंडी", "इल्ली", "सूंड़ी", "दीमक",
        "माहू", "चेपा", "तेला", "सफेद मक्खी", "ब्लास्ट", "खैरा", "सड़न", "उकठा", "रोकथाम", "इलाज", "उपचार", "नियंत्रण",
        "उपाय", "बचाव", "पत्ती मुड़ना", "पत्ता मरोड़", "लीफ कर्ल", "मरोड़िया",
        "ਕੀੜੇ", "ਬਿਮਾਰੀ", "ਸਪਰੇਅ", "ਕੀਟਨਾਸ਼ਕ", "ਕੁੰਗੀ", "ਤੇਲਾ", "ਝੁਲਸਾ", "ਸੁੰਡੀ", "ਇਲਾਜ", "ਰੋਕਥਾਮ",
        "পোকা", "রোগ", "কীটনাশক", "తెగులు", "పురుగు", "నివారణ", "பூச்சி", "நோய்", "જીવાત", "ઇયળ", "ರೋಗ", "ಹುಳು", "കീടങ്ങൾ", "کیڑے", "سپرے"
    ]
    if any(k in q for k in pest_keywords):
        return "pest_disease"

    # 4. Soil / Fertilizer / Nutrient Intent
    soil_keywords = [
        "soil", "fertilizer", "urea", "dap", "mop", "npk", "khad", "zinc", "potash", "nitrogen", "soil health", "soil test", "dose",
        "खाद", "उर्वरक", "यूरिया", "मिट्टी", "डीएपी", "पोटाश", "जिंक", "खत", "माती", "एनपीके", "गोबर", "खुराक", "मात्रा",
        "कब डालें", "कितना डालें", "अनुपात", "ਖਾਦ", "ਯੂਰੀਆ", "ਡੀਏਪੀ", "ਮਾਤਰਾ", "ਖੁਰਾਕ",
        "সার", "ইউরিয়া", "মাটি", "ఎరువు", "యూరియా", "మట్టి", "உரம்", "ખાતર", "ಗೊಬ್ಬರ", "വളം", "کھاد"
    ]
    if any(k in q for k in soil_keywords):
        return "soil"

    # 5. Irrigation / Watering Intent
    irrigation_keywords = [
        "irrigation", "water", "watering", "सिंचाई", "पहला पानी", "पानी", "सींच", "कब सींचें", "कब दें", "कब करें", "पिलाएं",
        "ਸਿੰਚਾਈ", "ਪਾਣੀ", "সেচ", "నీరు", "பாசனம்", "પિયત"
    ]
    if any(k in q for k in irrigation_keywords):
        return "crop"

    # 6. Climate Risk / Disaster Alert Intent
    risk_keywords = [
        "risk", "alert", "warning", "drought", "flood", "heat stress", "frost", "cold wave", "waterlogging",
        "जोखिम", "चेतावनी", "अलर्ट", "सूखा", "बाढ़", "पाला", "तुषार", "लू", "अतिवृष्टि", "ਧੋਕਾ", "ਖ਼ਤਰਾ", "ਚੇਤਾਵਨੀ", "ਸੋਕਾ"
    ]
    if any(k in q for k in risk_keywords):
        return "climate_risk"

    # 7. Crop Sowing / Variety / Cultivation Guide Intent
    crop_keywords = [
        "variety", "sowing", "seed", "crop", "beej", "plant", "yield", "season", "kharif", "rabi", "zaid",
        "किस्म", "बोने", "बीज", "पैदावार", "फसल", "रबी", "खरीफ", "उपज", "समय", "दूरी", "बुवाई", "कटाई",
        "ਕਿਸਮ", "ਬੀਜ", "ਫਸਲ", "ਰਬੀ", "ਬਿਜਾਈ", "রবী", "খরিফ", "విత్తనాలు", "పంట", "விதை", "பயிர்", "બિયારણ", "પાક", "ಬೆಳೆ", "വിള", "فصل"
    ]
    if any(k in q for k in crop_keywords):
        return "crop"

    # 8. Check if ANY crop name is mentioned directly
    common_crops = [
        "wheat", "rice", "cotton", "tomato", "potato", "mustard", "maize", "chickpea",
        "गेहूं", "धान", "चावल", "कपास", "टमाटर", "आलू", "सरसों", "मक्का", "चना",
        "ਕਣਕ", "ਝੋਨਾ", "ਕਪਾਹ", "ਟਮਾਟਰ", "ਸਰ੍ਹੋਂ", "ਗੰਨਾ", "sugarcane"
    ]
    if any(c in q for c in common_crops):
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
        "English": "Provide a comprehensive, ChatGPT-style detailed agronomic response with full depth. Cover Diagnosis, Organic IPM, Recommended Chemicals with exact dosages, Agronomic/Irrigation practices, and Expert Pro-Tips.",
        "Hindi": "चैटजीपीटी जैसा विस्तृत, संपूर्ण, स्पष्ट और वैज्ञानिक उत्तर दें। इसमें रोग/कीट पहचान, जैविक रोकथाम, अनुशंसित रसायनों की सटीक खुराक प्रति एकड़ व प्रति पंप, सस्य व सिंचाई प्रबंधन और वैज्ञानिक टिप्स शामिल करें।",
        "Punjabi": "ਪੂਰਾ ਵਿਸਤ੍ਰਿਤ, ਡੂੰਘਾ ਅਤੇ ਪੀਏਯੂ ਪ੍ਰਮਾਣਿਤ ਜਵਾਬ ਦਿਓ। ਸਮੱਸਿਆ ਦੀ ਪਛਾਣ, ਜੈਵਿਕ ਰੋਕਥਾਮ, ਰਸਾਇਣਕ ਦਵਾਈਆਂ ਦੀ ਸਹੀ ਖੁਰਾਕ, ਸਿੰਚਾਈ ਅਤੇ ਵਿਗਿਆਨਕ ਸਲਾਹ ਵਿਸਥਾਰ ਨਾਲ ਦੱਸੋ।",
        "Bengali": "চ্যাটজিপিটির মতো বিশদ এবং গভীর সমাধান দিন: পরিস্থিতি বিশ্লেষণ, জৈব প্রতিকার, রাসায়নিকের সঠিক মাত্রা, সেচ ব্যবস্থা ও কৃষি বিশেষজ্ঞদের টিপস।",
        "Telugu": "పూర్తి వివరాలతో కూడిన సమగ్ర శాస్త్రీయ సమాధానం ఇవ్వండి: సమస్య నిర్ధారణ, సేంద్రీయ పద్ధతులు, రసాయనాల ఖచ్చితమైన మోతాదులు, నీటి యాజమాన్యం మరియు నిపుణుల సలహాలు.",
        "Marathi": "सखोल आणि परिपूर्ण कृषी सल्ला द्या: रोगाची लक्षणे, जैविक उपाय, रासायनिक फवारणीचे अचूक प्रमाण, खत व पाणी व्यवस्थापन आणि तज्ज्ञ टिप्स.",
        "Tamil": "முழுமையான மற்றும் ஆழமான வேளாண்மை ஆலோசனையை வழங்கவும்: நோய் அடையாளம், இயற்கை முறைகள், சரியான ரசாயன மருந்தளவு, பாசன முறை மற்றும் விஞ்ஞானி குறிப்புகள்.",
        "Gujarati": "સંપૂર્ણ વિગતવાર વૈજ્ઞાનિક માર્ગદર્શન આપો: રોગની ઓળખ, જૈવિક ઉપચાર, ચોક્કસ રાસાયણિક ડોઝ, પિયત વ્યવસ્થા અને નિષ્ણાત સલાહ.",
        "Kannada": "ಸಂಪೂರ್ಣ ವಿವರವಾದ ಮತ್ತು ಆಳವಾದ ಕೃಷಿ ಸಲಹೆ ನೀಡಿ: ರೋಗ ಲಕ್ಷಣ, ಜೈವಿಕ ನಿಯಂತ್ರಣ, ರಾಸಾಯನಿಕಗಳ ನಿಖರ ಡೋಸ್, ನೀರಾವರಿ ಮತ್ತು ತಜ್ಞರ ಸಲಹೆಗಳು.",
        "Malayalam": "വിശദവും സമഗ്രവുമായ കാർഷിക നിർദ്ദേശങ്ങൾ നൽകുക: രോഗ ലക്ഷണം, ജൈവ രീതികൾ, കൃത്യമായ രാസ മരുന്നുകളുടെ അളവ്, നനയ്ക്കൽ ക്രമം, വിദഗ്ദ്ധ നിർദ്ദേശങ്ങൾ.",
        "Urdu": "جامع اور گہری زرعی رہنمائی فراہم کریں: مرض کی شناخت، قدرتی تدارک، کیمیائی ادویات کی درست خوراک، آبپاشی کا طریقہ اور زرعی ماہرین کے مشورے۔",
    }
    lang_instruction = lang_instructions.get(effective_language, lang_instructions["English"])

    full_prompt = (
        f"{system_prompt}\n\n"
        f"--- TARGETED FIELD DATA ---\n"
        f"{context}\n\n"
        f"--- FARMER QUERY ---\n"
        f"{augmented_message}\n\n"
        f"STRICT INSTRUCTION: {lang_instruction}\n"
        f"Provide a rich, deeply informative, beautifully formatted answer like ChatGPT. Ensure all technical dosages and schedules are accurate."
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
    # If no output matched primary slot, attempt fallback lookup across domain databases
    if not tool_outputs:
        p_fallback = get_pest_info(message)
        if "No specific pest data" not in p_fallback and "उपलब्ध नहीं" not in p_fallback:
            tool_outputs.append(p_fallback)
            sources.append("pest_disease_control")
        else:
            c_fallback = get_crop_recommendation(message)
            if "not found" not in c_fallback.lower():
                tool_outputs.append(c_fallback)
                sources.append("crop_recommendation")

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
