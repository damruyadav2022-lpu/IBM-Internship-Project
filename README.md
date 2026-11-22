# 🌾 KrishiRakshak (कृषि रक्षक • ਕ੍ਰਿਸ਼ੀ ਰੱਖਿਅਕ)
### Autonomous AI Agent for Smart Farming Advice, RAG & Climate Intelligence
**Problem Statement No. 9 — AI Agent for Smart Farming Advice**  
---

## 📖 1. About the Project

**KrishiRakshak** is an AI-powered agricultural decision-support platform designed for Indian smallholder farmers. Built to solve the grassroots agricultural information gap, KrishiRakshak integrates real-time hydro-meteorological intelligence, live APMC mandi wholesale commodity prices, deep-learning computer vision for plant pathology, and scientific soil health calibration based on **Punjab Agricultural University (PAU Ludhiana)** and **ICAR** research standards.

Farmers can interact naturally in their local language through typed text, spoken voice, or document/image uploads. The platform delivers sub-second, verified recommendations that reduce input costs, prevent disease outbreaks, and boost farm profitability.

---

## 🌟 2. Key Features & Tab Breakdown

The system is organized into **7 specialized, interactive modules**:

### 💬 Tab 1: AI Agricultural Advisor
* **What it does**: Answers open-ended agronomic questions about crop cycles, fertilizer dosing, weed management, and pest control.
* **Why it's needed**: Farmers often get generic, ungrounded advice from search engines or chemical retailers.
* **How it works**: Uses LangChain and Groq LLaMA 3.3 70B with a sub-second domain tool router and ChromaDB RAG knowledge base. Every answer includes a **"🔊 Listen"** voice playback button.

### 🎙️ Tab 2: Multilingual AI Voice Assistant
* **What it does**: A conversational speech-to-speech chatbot allowing non-literate farmers to ask questions verbally.
* **Why it's needed**: Eliminates typing and language barriers for rural farmers.
* **How it works**: Audio is transcribed via **Groq Whisper-large-v3** with automatic dialect identification; the agent synthesizes advice and outputs natural spoken audio via **gTTS**.

### 🛡️ Tab 3: Climate Risk Predictor & Early Warning System
* **What it does**: Forecasts climate risks (drought, flood, heat stress, frost, crop lodging) tailored to specific crop growth stages.
* **Why it's needed**: Sudden climate anomalies cause massive yield losses during critical stages (e.g., heatwaves during wheat grain filling).
* **How it works**: Ingests **ECMWF ERA5 Satellite Reanalysis** to track root-zone soil moisture ($0\text{--}7\text{ cm}$, $7\text{--}28\text{ cm}$, $28\text{--}100\text{ cm}$) and computes a **Climate Vulnerability Index (CVI, 0–100)** with what-if scenario simulation.

### 🔬 Tab 4: Plant Pathology (YOLOv8 Vision & PDF Diagnostic OCR)
* **What it does**: Diagnoses leaf diseases from photos, live camera snaps, or multi-page PDF diagnostic reports.
* **Why it's needed**: Early detection prevents crop failures.
* **How it works**: **YOLOv8** executes neural inference ($\approx 25\text{ ms}$) across 38 crop-disease classes. Features **"🎧 Spoken Diagnosis & Voice Output"**, providing biological treatments (Neem, Trichoderma), chemical sprays with exact dosages, and preventive cultural practices.

### 🧪 Tab 5: Soil Health Card (SHC) Scanner & PAU Fertilizer Planner
* **What it does**: Scans government Soil Health Cards, digitizes 12 parameters into an editable table, and generates a tailored PAU Ludhiana fertilizer prescription.
* **Why it's needed**: Prevents blind, excessive fertilizer usage, saving money and protecting soil fertility.
* **How it works**: Evaluates N, P, K, pH, EC, OC, S, Zn, Fe, Mn, B, and Cu. Omits redundant fertilizers (e.g., omitting MOP if soil potassium is high, saving ₹1,200+/acre) and prescribes a 3-split fertilizer schedule.

### 🌤️ Tab 6: Real-Time Weather & Agricultural Spray Windows
* **What it does**: Displays live weather conditions, 7-day agricultural forecasts, and spray-window alerts.
* **Why it's needed**: Spraying pesticides before rainfall or in high winds ($>15\text{ km/h}$) wastes costly chemicals.
* **How it works**: Connects directly to the **Open-Meteo ECMWF API** for precision hourly weather, humidity, and precipitation probability.

### 💰 Tab 7: Real-Time Mandi Prices (APMC Live Feeds & MSP)
* **What it does**: Displays wholesale prices across Indian APMC mandis, tracking trends and comparing them to Minimum Support Price (MSP) benchmarks.
* **Why it's needed**: Ensures farmers get fair prices and know whether to sell immediately or store their produce.
* **How it works**: Ingests official **Agmarknet** feeds with daily trend momentum (Rising ↑, Falling ↓, Stable →) and retail price equivalents.

---

## 🛠️ 3. Technology Stack

| Layer | Technologies | Purpose |
| :--- | :--- | :--- |
| **Frontend** | Streamlit, HTML5, Custom CSS | Modern dark-mode UI ("Midnight Forest"), audio players, and responsive data editors. |
| **Backend & API** | FastAPI, Pydantic v2, Uvicorn | High-performance asynchronous REST API gateway. |
| **AI / LLM** | Groq LLaMA 3.3 70B, LangChain | Sub-second agricultural reasoning and real-time data synthesis. |
| **RAG & Vector DB** | ChromaDB, Sentence-Transformers (`all-MiniLM-L6-v2`) | Embeds and retrieves verified ICAR and PAU agronomic knowledge. |
| **Computer Vision** | Ultralytics YOLOv8, PyPDFium2, Pillow | Deep-learning leaf classification and multi-page PDF report OCR. |
| **Speech Processing** | Groq Whisper-large-v3, gTTS | Multilingual Speech-to-Text (STT) and natural Text-to-Speech (TTS). |
| **Weather & Climate** | Open-Meteo, ECMWF ERA5 | High-resolution satellite root-zone moisture and weather forecasting. |
| **Market Data** | Agmarknet (Govt. of India) | Real-time APMC mandi wholesale rates and 2024–25 MSP data. |

---

## 🌐 4. Multilingual & Auto-Detection Engine

KrishiRakshak supports **11 languages**:
1. **English** (`en`)
2. **Hindi / हिन्दी** (`hi`)
3. **Punjabi / ਪੰਜਾਬੀ** (`pa`)
4. **Bengali / বাংলা** (`bn`)
5. **Telugu / తెలుగు** (`te`)
6. **Marathi / मराठी** (`mr`)
7. **Tamil / தமிழ்** (`ta`)
8. **Gujarati / ગુજરાતી** (`gu`)
9. **Kannada / ಕನ್ನಡ** (`kn`)
10. **Malayalam / മലയാളം** (`ml`)
11. **Urdu / اردو** (`ur`)

### ⚡ Zero-Latency Script Auto-Detection
Instead of heavy language-detection models, KrishiRakshak uses direct **Unicode codepoint ranges** ($\approx 0\text{ ms}$ latency):
* **Gurmukhi** (`\u0A00`–`\u0A7F`) $\rightarrow$ Punjabi
* **Devanagari** (`\u0900`–`\u097F`) $\rightarrow$ Hindi / Marathi
* **Bengali** (`\u0980`–`\u09FF`) $\rightarrow$ Bengali
* **Telugu** (`\u0C00`–`\u0C7F`) $\rightarrow$ Telugu
* **Tamil** (`\u0B80`–`\u0BFF`) $\rightarrow$ Tamil
* **Gujarati** (`\u0A80`–`\u0AFF`) $\rightarrow$ Gujarati
* **Kannada** (`\u0C80`–`\u0CFF`) $\rightarrow$ Kannada
* **Malayalam** (`\u0D00`–`\u0D7F`) $\rightarrow$ Malayalam
* **Perso-Arabic** (`\u0600`–`\u06FF`) $\rightarrow$ Urdu

---

## 📁 5. Folder & File Structure

```text
KrishiRakshak/
├── streamlit_app.py             # Main 7-tab Streamlit frontend
├── requirements.txt             # Project dependencies
├── .env                         # API keys (Groq key, base URL)
│
├── backend/
│   ├── main.py                  # FastAPI application entry point
│   ├── voice_assistant.py       # Whisper STT & gTTS audio engine
│   │
│   ├── agent/
│   │   ├── farming_agent.py     # LangChain multi-tool agent & prompt templates
│   │   ├── llm_client.py        # Groq LLaMA client initializer
│   │   ├── rag_pipeline.py      # ChromaDB vector retrieval pipeline
│   │   └── tools/               # Domain tool wrappers
│   │       ├── soil_tool.py
│   │       ├── climate_risk_tool.py
│   │       ├── weather_tool.py
│   │       ├── crop_tool.py
│   │       ├── pest_tool.py
│   │       ├── market_tool.py
│   │       └── disease_detection_tool.py
│   │
│   ├── api/
│   │   ├── models.py            # Pydantic data schemas
│   │   └── routes.py            # FastAPI REST endpoints
│   │
│   └── services/
│       ├── soil_health_service.py   # PAU Ludhiana 12-nutrient calibration engine
│       ├── climate_risk_service.py  # ERA5 satellite & CVI calculator
│       ├── weather_service.py       # Open-Meteo live observation service
│       ├── mandi_service.py         # Agmarknet APMC live market service
│       └── translation_service.py   # Centralized UI & disease translation catalog
│
├── data/
│   ├── crop_calendar.json       # Sowing and harvesting schedules
│   ├── disease_database.json    # 38 disease treatments and symptoms
│   ├── pest_database.json       # Biological & chemical pest solutions
│   ├── mandi_prices.json        # Benchmark APMC price records
│   ├── soil_conditions.json     # PAU Ludhiana fertilizer calibration tables
│   └── plantvillage/test/       # Sample leaf images for instant testing
│
└── scripts/
    ├── test_soil_health.py      # Automated tests for Soil Health & 7 tabs
    └── test_multilingual.py     # Automated tests for 11 languages, STT, and TTS
```

---

## 📦 6. Requirements & Dependencies

The project uses stable, well-maintained libraries listed in `requirements.txt`:

```text
# Web & UI
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
streamlit>=1.32.0
requests>=2.31.0
pandas>=2.1.4
pydantic>=2.5.3
pydantic-settings>=2.1.0
python-dotenv>=1.0.0
aiofiles>=23.2.1
python-multipart>=0.0.7

# AI, Agentic Reasoning & Vector Search
langchain>=0.1.5
langchain-community>=0.0.17
langchain-openai>=0.0.5
openai>=1.10.0
chromadb>=0.4.22
sentence-transformers>=2.3.1
httpx>=0.26.0

# Deep Learning & Computer Vision
torch>=2.1.0
torchvision>=0.16.0
ultralytics>=8.1.0
Pillow>=10.2.0
opencv-python-headless>=4.9.0
pypdfium2>=4.26.0

# Voice Processing
gTTS>=2.5.0
```

---

## 🚀 7. Installation & Quick Start

### Step 1: Clone the Repository
```bash
git clone https://github.com/damruyadav2022-lpu/IBM-Internship-Project.git
cd KrishiRakshak
```

### Step 2: Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows:
.\.venv\Scripts\activate

# Activate on Linux / macOS:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the root directory:
```env
LLM_API_KEY=gsk_your_groq_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
```
*(Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys))*

### Step 5: Launch the Application
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 8. Running Automated Tests

Run the included automated test suites to verify functionality:

```bash
# 1. Test Soil Health Card Scanner & 7-Tab Smoke Test
python scripts/test_soil_health.py

# 2. Test 11-Language Detection, Whisper STT, gTTS, and Agent Routing
python scripts/test_multilingual.py
```

---

## ☁️ 9. Deploying to Streamlit Community Cloud

1. Fork or push this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io) and click **"New app"**.
3. Select your repository, branch (`main`), and main file path (`streamlit_app.py`).
4. In **Advanced Settings → Secrets**, add:
   ```toml
   LLM_API_KEY = "gsk_..."
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   ```
5. Click **Deploy**.

---

## 💡 10. Frequently Asked Questions (Evaluator Cheatsheet)

**Q: Why not use standard ChatGPT directly?**  
*A*: Standard LLMs hallucinate chemical dosages, lack real-time APMC mandi prices, cannot access live Open-Meteo satellite feeds, and have no knowledge of regional soil calibration (such as **PAU Ludhiana's Package of Practices**). KrishiRakshak uses the LLM strictly as a natural-language synthesis layer grounded in verified domain microservices and RAG documents.

**Q: How does the system achieve sub-second response times?**  
*A*: Intent detection and entity extraction run through a lightweight heuristic router in $< 5\text{ ms}$, invoking local microservices directly. The Groq LLaMA 3.3 70B engine synthesizes the verified observation in $< 400\text{ ms}$, keeping the total pipeline execution well under $1\text{ second}$.

**Q: How does the Soil Health feature save farmers money?**  
*A*: Many farmers apply NPK fertilizers blindly. KrishiRakshak's PAU Ludhiana engine checks baseline soil potassium; if native potassium is high ($>280\text{ kg/ha}$), it completely omits MOP fertilizer, saving the farmer approximately ₹1,200 to ₹1,500 per acre in unnecessary input costs.

---

## 👤 Author
* **Developer**: Deepak Kumar Yadav
* **Email**: damruyadav2022@gmail.com
* **Repository**: [IBM-Internship-Project](https://github.com/damruyadav2022-lpu/IBM-Internship-Project.git)