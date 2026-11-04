# KrishiRakshak — Architecture Overview

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    KrishiRakshak System                      │
│                                                              │
│  ┌─────────────────┐         ┌──────────────────────────┐   │
│  │   React Frontend │ ──────► │   FastAPI Backend         │   │
│  │   (Port 3000)    │  HTTP   │   (Port 8000)             │   │
│  │                  │ ◄────── │                           │   │
│  │  • Chat UI       │  JSON   │  • /api/chat (POST)       │   │
│  │  • Language EN/HI│         │  • /api/health (GET)      │   │
│  │  • Quick Queries │         │  • /api/crops (GET)       │   │
│  │  • Source viewer │         │  • /api/markets (GET)     │   │
│  └─────────────────┘         └───────────┬──────────────┘   │
│                                           │                   │
│                               ┌───────────▼──────────────┐   │
│                               │   LangChain ReAct Agent   │   │
│                               │                           │   │
│                               │  Tools:                   │   │
│                               │  • weather_forecast       │   │
│                               │  • crop_recommendation    │   │
│                               │  • pest_disease_control   │   │
│                               │  • mandi_price_checker    │   │
│                               │  • agricultural_kb (RAG)  │   │
│                               └───────┬───────────────────┘   │
│                                       │                        │
│               ┌───────────────────────┼───────────────┐       │
│               │                       │               │       │
│    ┌──────────▼────┐    ┌─────────────▼──┐  ┌────────▼────┐  │
│    │  ChromaDB      │    │  LLM Client     │  │ JSON Data   │  │
│    │  Vector Store  │    │  (OpenAI-compat) │  │ Files       │  │
│    │                │    │                 │  │             │  │
│    │  Embeddings:   │    │  Providers:     │  │ weather_    │  │
│    │  all-MiniLM-   │    │  • OpenAI       │  │ data.json   │  │
│    │  L6-v2         │    │  • Ollama       │  │ crop_       │  │
│    │                │    │  • HuggingFace  │  │ calendar.   │  │
│    │  Collections:  │    │  • LM Studio    │  │ json        │  │
│    │  krishirakshak │    │  • Together AI  │  │ pest_       │  │
│    └───────────────┘    └─────────────────┘  │ control.    │  │
│                                               │ json        │  │
│                                               │ mandi_      │  │
│                                               │ prices.json │  │
│                                               │ soil_       │  │
│                                               │ conditions. │  │
│                                               │ json        │  │
│                                               └─────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### Frontend (React.js + Tailwind CSS)

| Component | File | Description |
|---|---|---|
| App | `App.js` | Root component, manages language and location state |
| ChatInterface | `ChatInterface.js` | Main chat window with message history, send/clear |
| MessageBubble | `MessageBubble.js` | Renders user/assistant bubbles with Markdown and source tags |
| QuickQuestions | `QuickQuestions.js` | Preset farming queries in English and Hindi |
| LanguageSelector | `LanguageSelector.js` | Toggle between English and Hindi |
| API Service | `services/api.js` | Axios wrapper for backend communication |

### Backend (FastAPI + LangChain)

| Module | File | Description |
|---|---|---|
| App Entry | `main.py` | FastAPI app, CORS, lifespan startup ingestion |
| Agent | `agent/farming_agent.py` | LangChain ReAct agent orchestrator |
| RAG Pipeline | `agent/rag_pipeline.py` | ChromaDB + sentence-transformers setup |
| LLM Client | `agent/llm_client.py` | OpenAI-compatible LLM abstraction |
| Weather Tool | `agent/tools/weather_tool.py` | Reads `weather_data.json` |
| Crop Tool | `agent/tools/crop_tool.py` | Reads `crop_calendar.json` |
| Pest Tool | `agent/tools/pest_tool.py` | Reads `pest_control.json` |
| Market Tool | `agent/tools/market_tool.py` | Reads `mandi_prices.json` |
| API Routes | `api/routes.py` | FastAPI route handlers |
| API Models | `api/models.py` | Pydantic request/response models |
| Data Loader | `utils/data_loader.py` | Ingests JSON files into ChromaDB |

## RAG Pipeline

```
JSON Data Files ──► DataLoader ──► Document Chunks ──► ChromaDB
                                                            │
User Question ─────────────────────────────────────────────▼
                                                    Similarity Search
                                                            │
                                                    Top-K Relevant Docs
                                                            │
                                                    LangChain Retriever
                                                            │
                                              LangChain ReAct Agent ◄── LLM
                                                            │
                                              Tool Selection & Execution
                                                            │
                                                    Final Answer
```

## Agent ReAct Loop

```
Question Input
      │
      ▼
[Thought] → Decide which tool to call
      │
      ▼
[Action] → Call tool (weather / crop / pest / market / RAG)
      │
      ▼
[Observation] → Parse tool output
      │
      ▼
Repeat if more tools needed (max 6 iterations)
      │
      ▼
[Final Answer] → Return to user
```

## Data Flow

```
Farmer Input (English/Hindi)
  └─► ChatInterface.js
       └─► POST /api/chat {message, language, location}
            └─► farming_agent.run_agent()
                 ├─► weather_tool → weather_data.json
                 ├─► crop_tool → crop_calendar.json
                 ├─► pest_tool → pest_control.json
                 ├─► market_tool → mandi_prices.json
                 └─► RAG tool → ChromaDB → sentence-transformers
                      └─► LLM (OpenAI / Ollama / HuggingFace)
                           └─► Final Answer + Sources + Agent Steps
                                └─► ChatResponse JSON
                                     └─► MessageBubble (rendered)
```

## LLM Configuration Options

| Provider | LLM_PROVIDER | LLM_MODEL | LLM_BASE_URL |
|---|---|---|---|
| OpenAI | `openai` | `gpt-3.5-turbo` | _(not needed)_ |
| OpenAI GPT-4 | `openai` | `gpt-4o-mini` | _(not needed)_ |
| Ollama (local) | `ollama` | `llama2` / `mistral` | `http://localhost:11434/v1` |
| LM Studio | `openai` | _(your model)_ | `http://localhost:1234/v1` |
| HuggingFace TGI | `huggingface` | `mistralai/Mistral-7B-Instruct-v0.2` | HF endpoint URL |
| Together AI | `openai` | `mistralai/Mixtral-8x7B` | `https://api.together.xyz/v1` |
| Groq | `openai` | `llama3-8b-8192` | `https://api.groq.com/openai/v1` |

## ChromaDB Collections

The vector store uses a single collection `krishirakshak` containing documents from all five data sources:

| Source | Document Count (approx.) | Metadata Fields |
|---|---|---|
| Weather Data | ~30 | city, date, type=weather |
| Crop Calendar | ~20 | crop, season, type=crop |
| Pest Control | ~15 | crop, pest, type=pest |
| Mandi Prices | ~55 | market, commodity, type=market |
| Soil Conditions | ~10 | region, soil_type, type=soil |

## API Endpoints

### POST /api/chat
```json
Request:
{
  "message": "What crop should I grow this Kharif season?",
  "language": "English",
  "location": "Delhi"
}

Response:
{
  "response": "For the Kharif season in Delhi...",
  "sources": ["crop_recommendation", "agricultural_knowledge_base"],
  "agent_steps": [
    {
      "tool": "crop_recommendation",
      "input": "Kharif season crops",
      "output": "Kharif Season Crops: Rice, Maize, Cotton..."
    }
  ],
  "language": "English",
  "location": "Delhi"
}
```

### GET /api/health
```json
{
  "status": "healthy",
  "service": "KrishiRakshak Backend",
  "version": "1.0.0"
}
```
