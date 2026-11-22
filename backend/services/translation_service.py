"""
Translation & Localization Service for KrishiRakshak.
Provides end-to-end multi-language support across all 7 tabs:
- UI string localization for 11 languages (English, Hindi, Punjabi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Urdu)
- Dynamic and dictionary-based agricultural translation
- Pathology diagnosis translation and spoken voice synthesis
- Climate risk advisory localization
"""
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SUPPORTED_LANGS = [
    "English", "Hindi", "Punjabi", "Bengali", "Telugu",
    "Marathi", "Tamil", "Gujarati", "Kannada", "Malayalam", "Urdu"
]

# ── 1. UI Strings across all 11 Languages ──────────────────────────────────────
UI_STRINGS: Dict[str, Dict[str, str]] = {
    "app_title": {
        "English": "KrishiRakshak — AI Agent for Smart Farming Advice",
        "Hindi": "कृषि रक्षक — स्मार्ट कृषि सलाह AI सहायक",
        "Punjabi": "ਕ੍ਰਿਸ਼ੀ ਰੱਖਿਅਕ — ਸਮਾਰਟ ਖੇਤੀਬਾੜੀ ਸਲਾਹ AI ਸਹਾਇਕ",
        "Bengali": "কৃষিরক্ষক — স্মার্ট কৃষি পরামর্শ এআই সহকারী",
        "Telugu": "కృషిరక్షక్ — స్మార్ట్ వ్యవసాయ సలహా AI అసిస్టెంట్",
        "Marathi": "कृषिरक्षक — स्मार्ट शेती सल्लागार AI सहाय्यक",
        "Tamil": "கிருஷிரக்ஷக் — ஸ்மார்ட் வேளாண்மை AI உதவியாளர்",
        "Gujarati": "કૃષિરક્ષક — સ્માર્ટ ખેતી સલાહ AI સહાયક",
        "Kannada": "ಕೃಷಿರಕ್ಷಕ್ — ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಸಲಹೆ AI ಸಹಾಯಕ",
        "Malayalam": "കൃഷിരക്ഷക് — സ്മാർട്ട് കാർഷിക AI സഹായി",
        "Urdu": "کرشی رکشاک — اسمارٹ زرعی مشاورتی AI معاون",
    },
    "tab_chat": {
        "English": "💬 AI Advisor",
        "Hindi": "💬 AI कृषि सलाहकार",
        "Punjabi": "💬 AI ਖੇਤੀ ਸਲਾਹਕਾਰ",
        "Bengali": "💬 এআই কৃষি পরামর্শদাতা",
        "Telugu": "💬 AI వ్యవసాయ సలహాదారు",
        "Marathi": "💬 AI कृषी सल्लागार",
        "Tamil": "💬 AI வேளாண் ஆலோசகர்",
        "Gujarati": "💬 AI કૃષિ સલાહકાર",
        "Kannada": "💬 AI ಕೃಷಿ ಸಲಹೆಗಾರ",
        "Malayalam": "💬 AI കാർഷിക ഉപദേശകൻ",
        "Urdu": "💬 AI زرعی مشیر",
    },
    "tab_voice": {
        "English": "🎙️ AI Voice Assistant",
        "Hindi": "🎙️ AI वॉइस सहायक (बोलकर पूछें)",
        "Punjabi": "🎙️ AI ਆਵਾਜ਼ ਸਹਾਇਕ (ਬੋਲ ਕੇ ਪੁੱਛੋ)",
        "Bengali": "🎙️ এআই ভয়েস সহকারী",
        "Telugu": "🎙️ AI వాయిస్ అసిస్టెంట్",
        "Marathi": "🎙️ AI व्हॉईस सहाय्यक",
        "Tamil": "🎙️ AI குரல் உதவியாளர்",
        "Gujarati": "🎙️ AI વૉઇસ સહાયક",
        "Kannada": "🎙️ AI ಧ್ವನಿ ಸಹಾಯಕ",
        "Malayalam": "🎙️ AI വോയ്‌സ് അസിസ്റ്റന്റ്",
        "Urdu": "🎙️ AI صوتی معاون",
    },
    "tab_risk": {
        "English": "🛡️ Climate Risk & Alerts",
        "Hindi": "🛡️ मौसम जोखिम व चेतावनी",
        "Punjabi": "🛡️ ਮੌਸਮ ਖ਼ਤਰਾ ਤੇ ਚੇਤਾਵਨੀ",
        "Bengali": "🛡️ জলবায়ু ঝুঁকি ও সতর্কতা",
        "Telugu": "🛡️ వాతావరణ ముప్పు & హెచ్చరికలు",
        "Marathi": "🛡️ हवामान धोका व पूर्वसूचना",
        "Tamil": "🛡️ காலநிலை எச்சரிக்கைகள்",
        "Gujarati": "🛡️ વાતાવરણ જોખમ અને ચેતવણી",
        "Kannada": "🛡️ ಹವಾಮಾನ ಅಪಾಯ ಮತ್ತು ಎಚ್ಚರಿಕೆ",
        "Malayalam": "🛡️ കാലാവസ്ഥാ മുന്നറിയിപ്പ്",
        "Urdu": "🛡️ موسمی خطرات اور انتباہ",
    },
    "tab_disease": {
        "English": "🔬 Plant Pathology (YOLOv8)",
        "Hindi": "🔬 फसल रोग पहचान व उपचार",
        "Punjabi": "🔬 ਪੌਦਾ ਬਿਮਾਰੀ ਜਾਂਚ ਤੇ ਇਲਾਜ",
        "Bengali": "🔬 ফসলের রোগ নির্ণয় ও চিকিৎসা",
        "Telugu": "🔬 పంట తెగుళ్ల గుర్తింపు & నివారణ",
        "Marathi": "🔬 पीक रोग निदान व उपाय",
        "Tamil": "🔬 பயிர் நோய் கண்டறிதல்",
        "Gujarati": "🔬 પાક રોગ નિદાન અને સારવાર",
        "Kannada": "🔬 ಸಸ್ಯ ರೋಗ ಪತ್ತೆ ಮತ್ತು ಚಿಕಿತ್ಸೆ",
        "Malayalam": "🔬 സസ്യരോഗ നിർണ്ണയം",
        "Urdu": "🔬 پودوں کی بیماریوں کی تشخیص",
    },
    "tab_soil": {
        "English": "🧪 Soil Health & Fertilizer (PAU)",
        "Hindi": "🧪 सॉइल हेल्थ व खाद कैलकुलेटर (PAU)",
        "Punjabi": "🧪 ਮਿੱਟੀ ਪਰਖ ਤੇ ਖਾਦ ਯੋਜਨਾ (PAU)",
        "Bengali": "🧪 মাটি পরীক্ষা ও সুষম সার",
        "Telugu": "🧪 నేల పరీక్ష & ఎరువుల ప్రణాళిక",
        "Marathi": "🧪 माती परीक्षण व खत नियोजन",
        "Tamil": "🧪 மண் பரிசோதனை & உரம்",
        "Gujarati": "🧪 જમીન ચકાસણી અને ખાતર",
        "Kannada": "🧪 ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಮತ್ತು ಗೊಬ್ಬರ",
        "Malayalam": "🧪 മണ്ണ് പരിശോധന & വളം",
        "Urdu": "🧪 مٹی کی جانچ اور کھاد کا منصوبہ",
    },
    "tab_weather": {
        "English": "🌤️ Real-Time Weather",
        "Hindi": "🌤️ रीयल-टाइम मौसम",
        "Punjabi": "🌤️ ਤਾਜ਼ਾ ਮੌਸਮ ਜਾਣਕਾਰੀ",
        "Bengali": "🌤️ রিয়েল-টাইম আবহাওয়া",
        "Telugu": "🌤️ ప్రత్యక్ష వాతావరణం",
        "Marathi": "🌤️ थेट हवामान अंदाज",
        "Tamil": "🌤️ நேரலை வானிலை",
        "Gujarati": "🌤️ લાઈવ હવામાન",
        "Kannada": "🌤️ ನೇರ ಹವಾಮಾನ",
        "Malayalam": "🌤️ തത്സമയ കാലാവസ്ഥ",
        "Urdu": "🌤️ لائیو موسم",
    },
    "tab_mandi": {
        "English": "💰 Mandi Prices (APMC Live)",
        "Hindi": "💰 लाइव मंडी भाव (APMC)",
        "Punjabi": "💰 ਤਾਜ਼ਾ ਮੰਡੀ ਭਾਅ (APMC)",
        "Bengali": "💰 লাইভ বাজার দর (মন্ডি)",
        "Telugu": "💰 ప్రత్యక్ష మార్కెట్ ధరలు",
        "Marathi": "💰 थेट बाजारभाव (मंडी)",
        "Tamil": "💰 நேரலை மண்டி விலை",
        "Gujarati": "💰 લાઈવ માર્કેટ ભાવ",
        "Kannada": "💰 ಮಾರುಕಟ್ಟೆ ಧಾರಣೆ",
        "Malayalam": "💰 വിപണി വിലനിലവാരം",
        "Urdu": "💰 لائیو منڈی ریٹ",
    },
    "spoken_diagnosis_title": {
        "English": "🎧 Spoken Diagnosis & Voice Output",
        "Hindi": "🎧 बोलकर रोग निदान व ऑडियो सलाह",
        "Punjabi": "🎧 ਬੋਲ ਕੇ ਬਿਮਾਰੀ ਜਾਂਚ ਤੇ ਆਡੀਓ ਸਲਾਹ",
        "Bengali": "🎧 ভয়েস ডায়াগনসিস ও অডিও প্রেসক্রিপশন",
        "Telugu": "🎧 వాయిస్ రోగ నిర్ధారణ & ఆడియో సలహా",
        "Marathi": "🎧 व्हॉईस रोग निदान व ऑडिओ सल्ला",
        "Tamil": "🎧 குரல் வழி நோய் கண்டறிதல்",
        "Gujarati": "🎧 વૉઇસ રોગ નિદાન અને ઑડિયો સલાહ",
        "Kannada": "🎧 ಧ್ವನಿ ರೋಗ ಪತ್ತೆ ಮತ್ತು ಆಡಿಯೋ ಸಲಹೆ",
        "Malayalam": "🎧 വോയ്‌സ് രോഗനിർണ്ണയവും ഓഡിയോയും",
        "Urdu": "🎧 صوتی تشخیص اور آڈیو ایڈوائزری",
    },
    "spoken_diagnosis_caption": {
        "English": "Listen to natural spoken diagnosis, biological remedies, and chemical spray timings in your language.",
        "Hindi": "अपनी भाषा में रोग के लक्षण, जैविक उपचार और रासायनिक छिड़काव की प्राकृतिक ऑडियो सलाह सुनें।",
        "Punjabi": "ਆਪਣੀ ਬੋਲੀ ਵਿੱਚ ਬਿਮਾਰੀ ਦੇ ਲੱਛਣ, ਦੇਸੀ ਇਲਾਜ ਅਤੇ ਸਪਰੇਅ ਦੀ ਜਾਣਕਾਰੀ ਸੁਣੋ।",
        "Bengali": "আপনার ভাষায় রোগের লক্ষণ, জৈব প্রতিকার ও রাসায়নিক স্প্রে সংক্রান্ত অডিও শুনুন।",
        "Telugu": "మీ స్వంత భాషలో వ్యాధి లక్షణాలు, సేంద్రీయ నివారణ మరియు స్ప్రే సమయాలను వినండి.",
        "Marathi": "तुमच्या भाषेत रोगाची लक्षणे, सेंद्रिय उपाय व फवारणीचे वेळापत्रक ऐका.",
        "Tamil": "உங்கள் மொழியில் நோய் மேலாண்மை மற்றும் தெளிப்பு பரிந்துரைகளைக் கேளுங்கள்.",
        "Gujarati": "તમારી ભાષામાં રોગના લક્ષણો અને ઉપચારની ઑડિયો સલાહ સાંભળો.",
        "Kannada": "ನಿಮ್ಮ ಭಾಷೆಯಲ್ಲಿ ರೋಗದ ಲಕ್ಷಣಗಳು ಮತ್ತು ಚಿಕಿತ್ಸೆಯ ಆಡಿಯೋ ಸಲಹೆಯನ್ನು ಕೇಳಿ.",
        "Malayalam": "നിങ്ങളുടെ ഭാഷയിൽ രോഗലക്ഷണങ്ങളും പ്രതിവിധികളും കേൾക്കുക.",
        "Urdu": "اپنی زبان میں بیماری کے علاج اور سپرے کی ہدایات سنیں۔",
    },
    "listen_button": {
        "English": "🔊 Listen Spoken Audio",
        "Hindi": "🔊 बोलकर सुनें (ऑडियो)",
        "Punjabi": "🔊 ਆਵਾਜ਼ ਸੁਣੋ (ਆਡੀਓ)",
        "Bengali": "🔊 অডিও শুনুন",
        "Telugu": "🔊 ఆడియో వినండి",
        "Marathi": "🔊 ऑडिओ ऐका",
        "Tamil": "🔊 ஆடியோ கேளுங்கள்",
        "Gujarati": "🔊 ઑડિયો સાંભળો",
        "Kannada": "🔊 ಆಡಿಯೋ ಕೇಳಿ",
        "Malayalam": "🔊 ഓഡിയോ കേൾക്കൂ",
        "Urdu": "🔊 آڈیو سنیں",
    },
    "active_region": {
        "English": "Active Region",
        "Hindi": "सक्रिय क्षेत्र",
        "Punjabi": "ਸਰਗਰਮ ਇਲਾਕਾ",
        "Bengali": "সক্রিয় অঞ্চল",
        "Telugu": "ప్రస్తుత ప్రాంతం",
        "Marathi": "सक्रिय प्रदेश",
        "Tamil": "செயலில் உள்ள பகுதி",
        "Gujarati": "સક્રિય પ્રદેશ",
        "Kannada": "ಸಕ್ರಿಯ ಪ್ರದೇಶ",
        "Malayalam": "പ്രദേശം",
        "Urdu": "فعال علاقہ",
    },
    "organic_treatment": {
        "English": "🍃 Organic & Biological Control",
        "Hindi": "🍃 जैविक व प्राकृतिक रोकथाम",
        "Punjabi": "🍃 ਜੈਵਿਕ ਤੇ ਦੇਸੀ ਰੋਕਥਾਮ",
        "Bengali": "🍃 জৈব ও প্রাকৃতিক প্রতিকার",
        "Telugu": "🍃 సేంద్రీయ & జీవ నియంత్రణ",
        "Marathi": "🍃 सेंद्रिय व जैविक उपाय",
        "Tamil": "🍃 இயற்கை பாதுகாப்பு",
        "Gujarati": "🍃 જૈવિક અને કુદરતી નિયંત્રણ",
        "Kannada": "🍃 ಸಾವಯವ ನಿಯಂತ್ರಣ",
        "Malayalam": "🍃 ജൈവ നിയന്ത്രണം",
        "Urdu": "🍃 حیاتیاتی اور قدرتی علاج",
    },
    "chemical_spray": {
        "English": "💊 Recommended Chemical Spray",
        "Hindi": "💊 अनुशंसित रासायनिक छिड़काव",
        "Punjabi": "💊 ਸਿਫ਼ਾਰਸ਼ ਕੀਤੀ ਰਸਾਇਣਕ ਸਪਰੇਅ",
        "Bengali": "💊 অনুমোদিত রাসায়নিক স্প্রে",
        "Telugu": "💊 సిఫార్సు చేసిన రసాయన స్ప్రే",
        "Marathi": "💊 रासायनिक फवारणी शिफारस",
        "Tamil": "💊 பரிந்துரைக்கப்பட்ட தெளிப்பு",
        "Gujarati": "💊 રાસાયણિક છંટકાવ",
        "Kannada": "💊 ಶಿಫಾರಸು ಮಾಡಿದ ರಾಸಾಯನಿಕ ಸಿಂಪಡಣೆ",
        "Malayalam": "💊 രാസ തളിക്കൽ",
        "Urdu": "💊 تجویز کردہ کیمیائی سپرے",
    },
    "prevention": {
        "English": "🛡️ Preventive Agronomic Measures",
        "Hindi": "🛡️ रोग से बचाव के कृषि उपाय",
        "Punjabi": "🛡️ ਬਚਾਅ ਸੰਬੰਧੀ ਖੇਤੀਬਾੜੀ ਸਾਵਧਾਨੀਆਂ",
        "Bengali": "🛡️ আগাম প্রতিরোধমূলক ব্যবস্থা",
        "Telugu": "🛡️ ముందస్తు నివారణ చర్యలు",
        "Marathi": "🛡️ प्रतिबंधात्मक कृषी उपाय",
        "Tamil": "🛡️ தடுப்பு நடவடிக்கைகள்",
        "Gujarati": "🛡️ અગમચેતીના પગલાં",
        "Kannada": "🛡️ ಮುನ್ನೆಚ್ಚರಿಕೆ ಕ್ರಮಗಳು",
        "Malayalam": "🛡️ മുൻകരുതൽ നടപടികൾ",
        "Urdu": "🛡️ حفاظتی تدابیر",
    },
}

# ── 2. Localized Disease Knowledge Base ─────────────────────────────────────────
DISEASE_LOCALIZATION: Dict[str, Dict[str, Dict[str, str]]] = {
    "Tomato Early Blight": {
        "name": {
            "Hindi": "टमाटर का अगेती झुलसा (Early Blight)",
            "Punjabi": "ਟਮਾਟਰ ਦਾ ਅਗੇਤਾ ਝੁਲਸਾ ਰੋਗ",
            "Bengali": "টমেটোর আর্লি ব্লাইট (আগাম ধ্বসা)",
            "Telugu": "టమాటా ముందస్తు తెగులు (Early Blight)",
            "Marathi": "टोमॅटोवरील अगेती करपा",
            "Tamil": "தக்காளி ஆரம்பக்கால கருகல் நோய்",
            "Gujarati": "ટમેટાનો આગોતરો સુકારો",
            "Kannada": "ಟೊಮೆಟೊ ಆರಂಭಿಕ ಅಂಗಮಾರಿ ರೋಗ",
            "Malayalam": "തക്കാളിയുടെ ഏർലി ബ്ലൈറ്റ് രോഗം",
            "Urdu": "ٹماٹر کا اگیتی جھلساؤ",
        },
        "spoken_summary": {
            "Hindi": "टमाटर की फसल में अगेती झुलसा रोग के लक्षण दिखे हैं। तुरंत 2 मिली प्रति लीटर पानी में मैंकोजेब या कॉपर ऑक्सीक्लोराइड मिलाकर छिड़काव करें। पत्तों पर नीचे से पानी न छिड़कें।",
            "Punjabi": "ਟਮਾਟਰ ਦੀ ਫਸਲ ਵਿੱਚ ਅਗੇਤੇ ਝੁਲਸੇ ਦੇ ਲੱਛਣ ਮਿਲੇ ਹਨ। ਰੋਕਥਾਮ ਲਈ 2 ਗ੍ਰਾਮ ਪ੍ਰਤੀ ਲੀਟਰ ਪਾਣੀ ਵਿੱਚ ਮੈਨਕੋਜ਼ੇਬ ਜਾਂ ਕਾਪਰ ਆਕਸੀਕਲੋਰਾਈਡ ਘੋਲ ਕੇ ਤੁਰੰਤ ਛਿੜਕਾਅ ਕਰੋ।",
            "Bengali": "টমেটো গাছে আর্লি ব্লাইট রোগের লক্ষণ দেখা দিয়েছে। প্রতি লিটার পানিতে ২ গ্রাম ম্যানকোজেব গুলে স্প্রে করুন। আক্রান্ত পাতা কেটে নষ্ট করে ফেলুন।",
            "Telugu": "టమాటా పంటలో ముందస్తు తెగులు కనిపించింది. నివారణకు లీటరు నీటికి 2 గ్రాముల మాంకోజెబ్ లేదా కాపర్ ఆక్సీక్లోరైడ్ కలిపి పిచికారీ చేయండి.",
            "Marathi": "टोमॅटो पिकावर अगेती करपा रोगाची लक्षणे आढळली आहेत. नियंत्रणासाठी प्रति लिटर पाण्यात २ ग्रॅम मॅन्कोझेब मिसळून तातडीने फवारणी करावी.",
            "Tamil": "தக்காளியில் ஆரம்பக்கால கருகல் நோய் காணப்படுகிறது. ஒரு லிட்டர் தண்ணீருக்கு 2 கிராம் மேன்கோசெப் கலந்து தெளிக்கவும்.",
            "English": "Early Blight detected on tomato foliage. Spray Mancozeb (2g/L) or Copper Oxychloride immediately. Ensure good aeration and remove lower infected leaves.",
        }
    },
    "Tomato Late Blight": {
        "name": {
            "Hindi": "टमाटर का पछेती झुलसा (Late Blight)",
            "Punjabi": "ਟਮਾਟਰ ਦਾ ਪਛੇਤਾ ਝੁਲਸਾ ਰੋਗ",
            "Bengali": "টমেটোর নাবী ধ্বসা (লেট ব্লাইট)",
            "Telugu": "టమాటా లేట్ బ్లైట్ తెగులు",
            "Marathi": "टोमॅटोवरील पछेती करपा",
            "English": "Tomato Late Blight",
        },
        "spoken_summary": {
            "Hindi": "टमाटर में पछेती झुलसा एक गंभीर फफूंद रोग है। तुरंत रिडोमिल गोल्ड या साइमोक्सानिल का छिड़काव करें और खेत में जलभराव न होने दें।",
            "Punjabi": "ਟਮਾਟਰ ਵਿੱਚ ਪਛੇਤਾ ਝੁਲਸਾ ਫੈਲ ਰਿਹਾ ਹੈ। ਬਚਾਅ ਲਈ ਰਿਡੋਮਿਲ ਗੋਲਡ ਜਾਂ ਕੈਬਰੀਓ ਟਾਪ ਦਾ ਤੁਰੰਤ ਛਿੜਕਾਅ ਕਰੋ।",
            "Bengali": "টমেটোর লেট ব্লাইট খুব দ্রুত ছড়ায়। রিডোমিল গোল্ড বা মেটালাক্সিল গ্রুপের ছত্রাকনাশক স্প্রে করুন।",
            "Telugu": "టమాటా లేట్ బ్లైట్ తెగులు నివారణకు రిడోమిల్ గోల్డ్ పిచికారీ చేయండి మరియు నీరు నిల్వ ఉండకుండా చూడండి.",
            "Marathi": "टोमॅटो पिकावरील पछेती करपा नियंत्रणासाठी रिडोमिल गोल्डची फवारणी करावी.",
            "English": "Late Blight is aggressive. Apply Ridomil Gold (2.5g/L) or Cymoxanil immediately and maintain field drainage.",
        }
    },
    "Tomato Yellow Leaf Curl Virus": {
        "name": {
            "Hindi": "टमाटर का पत्ता मरोड़ विषाणु (Yellow Leaf Curl)",
            "Punjabi": "ਟਮਾਟਰ ਦਾ ਪੱਤਾ ਮਰੋੜ ਵਿਸ਼ਾਣੂ (ਕੁਰਲ ਵਾਇਰਸ)",
            "Bengali": "টমেটোর পাতা কোঁকড়ানো ভাইরাস",
            "Telugu": "టమాటా ఆకు ముడత వైరస్",
            "Marathi": "टोमॅटोवरील चुरडा-मुरडा (व्हायरस)",
            "English": "Tomato Yellow Leaf Curl Virus",
        },
        "spoken_summary": {
            "Hindi": "पत्ता मरोड़ रोग सफेद मक्खी कीट द्वारा फैलता है। सफेद मक्खी की रोकथाम के लिए इमिडाक्लोप्रिड या नीम तेल 5 मिली प्रति लीटर पानी में मिलाकर छिड़कें।",
            "Punjabi": "ਪੱਤਾ ਮਰੋੜ ਰੋਗ ਚਿੱਟੀ ਮੱਖੀ ਰਾਹੀਂ ਫੈਲਦਾ ਹੈ। ਚਿੱਟੀ ਮੱਖੀ ਰੋਕਣ ਲਈ ਇਮੀਡਾਕਲੋਪ੍ਰਿਡ ਜਾਂ ਨਿੰਮ ਦਾ ਤੇਲ ਸਪਰੇਅ ਕਰੋ।",
            "Bengali": "পাতা কোঁকড়ানো রোগ সাদা মাছির মাধ্যমে ছড়ায়। ইমিডাক্লোপ্রিড বা নিম তেল স্প্রে করে সাদা মাছি দমন করুন।",
            "Telugu": "ఆకు ముడత తెగులు తెల్లదోమ ద్వారా వ్యాపిస్తుంది. ఇమిడాక్లోప్రిడ్ లేదా వేప నూనెను పిచికారీ చేయండి.",
            "Marathi": "हा रोग पांढऱ्या माशीमुळे पसरतो. पांढरी माशी नियंत्रणासाठी इमिडाक्लोप्रिडची फवारणी करावी.",
            "English": "Vector whitefly transmits Tomato Yellow Leaf Curl. Spray Imidacloprid (0.5ml/L) or 5% Neem Seed Kernel Extract.",
        }
    },
    "Potato Early Blight": {
        "name": {
            "Hindi": "आलू का अगेती झुलसा",
            "Punjabi": "ਆਲੂਆਂ ਦਾ ਅਗੇਤਾ ਝੁਲਸਾ",
            "Bengali": "আলুর আর্লি ব্লাইট রোগ",
            "Telugu": "బంగాళాదుంప ముందస్తు తెగులు",
            "Marathi": "बटाट्यावरील अगेती करपा",
            "English": "Potato Early Blight",
        },
        "spoken_summary": {
            "Hindi": "आलू के पत्तों पर छल्लेदार भूरे धब्बे अगेती झुलसा के हैं। मैंकोजेब 2.5 ग्राम प्रति लीटर छिड़कें और पोटाश खाद संतुलित दें।",
            "Punjabi": "ਆਲੂਆਂ ਤੇ ਅਗੇਤੇ ਝੁਲਸੇ ਦੀ ਰੋਕਥਾਮ ਲਈ ਮੈਨਕੋਜ਼ੇਬ ਢਾਈ ਗ੍ਰਾਮ ਪ੍ਰਤੀ ਲੀਟਰ ਪਾਣੀ ਵਿੱਚ ਮਿਲਾ ਕੇ ਛਿੜਕੋ।",
            "Bengali": "আলুর আগাম ধ্বসা রোধে ম্যানকোজেব স্প্রে করুন এবং জমিতে পরিমিত ইউরিয়া ও পটাশ দিন।",
            "Telugu": "బంగాళాదుంప ఆకులపై మచ్చల నివారణకు మాంకోజెబ్ పిచికారీ చేయండి.",
            "Marathi": "बटाटा पिकावर अगेती करपा नियंत्रणासाठी मॅन्कोझेब २.५ ग्रॅम फवारावे.",
            "English": "Concentric rings on potato leaves indicate Early Blight. Apply Mancozeb 75 WP at 2.5g/L water.",
        }
    },
    "Corn Common Rust": {
        "name": {
            "Hindi": "मक्का का सामान्य रतुआ (रस्ट रोग)",
            "Punjabi": "ਮੱਕੀ ਦਾ ਰਤੂਆ (ਕੁੰਗੀ ਰੋਗ)",
            "Bengali": "ভুট্টার মরিচা রোগ (কমন রাস্ট)",
            "Telugu": "మొక్కజొన్న తుప్పు తెగులు (రస్ట్)",
            "Marathi": "मक्यावरील तांबेरा रोग",
            "English": "Corn Common Rust",
        },
        "spoken_summary": {
            "Hindi": "मक्का के पत्तों पर भूरे फफोले रतुआ रोग हैं। रोकथाम के लिए अज़ोक्सीस्ट्रोबिन या मैंकोजेब का छिड़काव करें।",
            "Punjabi": "ਮੱਕੀ ਦੇ ਪੱਤਿਆਂ ਤੇ ਕੁੰਗੀ ਦੇ ਨਿਸ਼ਾਨ ਹਨ। ਅਜ਼ੌਕਸੀਸਟ੍ਰੋਬਿਨ ਜਾਂ ਪ੍ਰੋਪੀਕੋਨਾਜ਼ੋਲ ਦੀ ਸਪਰੇਅ ਕਰੋ।",
            "Bengali": "ভুট্টার পাতায় মরিচা রোগের জন্য প্রপিকোনাজল বা ম্যানকোজেব স্প্রে করুন।",
            "Telugu": "మొక్కజొన్న తుప్పు తెగులు నివారణకు ప్రొపికోనజోల్ పిచికారీ చేయండి.",
            "Marathi": "मक्यावरील तांबेरा नियंत्रणासाठी प्रोपिकोनाझोल फवारावे.",
            "English": "Pustules on maize leaves represent Common Rust. Spray Azoxystrobin (1ml/L) or Propiconazole (1ml/L).",
        }
    },
}


def get_ui_text(key: str, lang: str = "English") -> str:
    """Retrieve localized UI string with automatic English fallback."""
    if not lang or lang.lower() in ("auto", "auto-detect"):
        lang = "English"

    # Match clean lang name
    clean_name = "English"
    for ln in SUPPORTED_LANGS:
        if ln.lower() in lang.lower():
            clean_name = ln
            break

    sub_dict = UI_STRINGS.get(key, {})
    return sub_dict.get(clean_name, sub_dict.get("English", key))


def get_localized_disease_name(display_name: str, lang: str = "English") -> str:
    """Return translated disease name for plant pathology."""
    clean_name = "English"
    for ln in SUPPORTED_LANGS:
        if ln.lower() in lang.lower():
            clean_name = ln
            break

    for d_key, info in DISEASE_LOCALIZATION.items():
        if d_key.lower() in display_name.lower():
            return info.get("name", {}).get(clean_name, display_name)
    return display_name


def get_spoken_diagnosis_text(display_name: str, lang: str = "English") -> str:
    """Return a natural, clear spoken audio diagnosis script for the farmer."""
    clean_name = "English"
    for ln in SUPPORTED_LANGS:
        if ln.lower() in lang.lower():
            clean_name = ln
            break

    for d_key, info in DISEASE_LOCALIZATION.items():
        if d_key.lower() in display_name.lower():
            summaries = info.get("spoken_summary", {})
            return summaries.get(clean_name, summaries.get("English", f"Pathology diagnosis: {display_name}. Consult local agricultural extension officer."))

    return f"Farming diagnosis for {display_name}: Apply certified fungicide or organic neem extract and avoid waterlogging."


def translate_text(text: str, target_lang: str = "English") -> str:
    """
    Translate agricultural text to target language using Groq LLM (sub-second)
    with offline phrase/keyword fallback.
    """
    if not text or not text.strip():
        return ""

    clean_target = "English"
    for ln in SUPPORTED_LANGS:
        if ln.lower() in target_lang.lower():
            clean_target = ln
            break

    if clean_target == "English":
        return text

    # Attempt ultra-fast Groq LLM translation
    try:
        from agent.llm_client import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = get_llm()
        if llm is not None:
            prompt = (
                f"You are a professional agricultural translator for Indian farmers. "
                f"Translate the following agricultural text into clear, simple {clean_target}. "
                f"Preserve chemical names (e.g. Mancozeb, Urea, DAP), numbers, and dosages accurately. "
                f"Return ONLY the translation without any commentary:\n\n{text}"
            )
            res = llm.invoke([
                SystemMessage(content=f"You are a direct agricultural translator to {clean_target}."),
                HumanMessage(content=prompt)
            ])
            trans = res.content.strip()
            if trans and len(trans) > 3:
                return trans
    except Exception as e:
        logger.warning(f"Groq dynamic translation fallback: {e}")

    # Fallback: check disease localization
    for d_key, info in DISEASE_LOCALIZATION.items():
        if d_key.lower() in text.lower():
            loc_name = info.get("name", {}).get(clean_target)
            if loc_name:
                return text.replace(d_key, loc_name)

    return text
