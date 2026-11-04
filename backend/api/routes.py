"""
API routes for KrishiRakshak.
POST /api/chat           — main agent endpoint
GET  /api/health         — health check
POST /api/detect-disease — YOLOv8 plant disease detection
GET  /api/diseases       — list supported disease classes
"""
import logging
from typing import Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.models import (
    AgentStep,
    ChatRequest,
    ChatResponse,
    DiseaseDetectionResponse,
    DetectionResult,
    HealthResponse,
)
from agent.farming_agent import run_agent

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main endpoint: send a farming question and receive AI-powered advice.

    The agent reasons over real-time weather, crop calendar, pest control,
    and mandi price data, augmented by RAG retrieval from the knowledge base.
    """
    logger.info(
        f"Chat request — location={request.location} "
        f"language={request.language} "
        f"message={request.message[:80]}..."
    )

    try:
        result = run_agent(
            message=request.message,
            language=request.language,
            location=request.location,
        )

        steps = [
            AgentStep(
                tool=step["tool"],
                input=str(step["input"]),
                output=step["output"],
            )
            for step in result.get("agent_steps", [])
        ]

        return ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            agent_steps=steps,
            language=request.language,
            location=request.location,
        )

    except Exception as e:
        logger.exception(f"Unexpected error in /api/chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}",
        )


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="KrishiRakshak Backend",
        version="1.0.0",
    )


@router.get("/crops", tags=["Data"])
async def list_crops():
    """Return a list of supported crops for UI dropdowns."""
    import json
    from pathlib import Path

    data_path = Path(__file__).parent.parent.parent / "data" / "crop_calendar.json"
    try:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        crops = []
        for season, season_data in data.get("seasons", {}).items():
            for crop in season_data.get("crops", []):
                crops.append({
                    "name": crop["name"],
                    "hindi_name": crop["hindi_name"],
                    "season": season,
                })
        return {"crops": crops}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/markets", tags=["Data"])
async def list_markets():
    """Return list of supported mandi locations and live date."""
    try:
        from services.mandi_service import get_mandi_service
        data = get_mandi_service().get_realtime_mandi_data()
        return {
            "markets": list(data.get("markets", {}).keys()),
            "date": data.get("date"),
            "is_live": True,
        }
    except Exception as e:
        logger.warning(f"Error fetching live markets list: {e}")
        import json
        from pathlib import Path
        data_path = Path(__file__).parent.parent.parent / "data" / "mandi_prices.json"
        try:
            with open(data_path, encoding="utf-8") as f:
                data = json.load(f)
            return {
                "markets": list(data.get("markets", {}).keys()),
                "date": data.get("date"),
                "is_live": False,
            }
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex))


@router.get("/weather", tags=["Real-Time Data"])
async def get_live_weather(location: str = "Delhi"):
    """
    Fetch real-time live weather, current conditions, and 7-day agricultural forecasts
    via Open-Meteo with dynamic agronomic advisories.
    """
    try:
        from services.weather_service import get_weather_service
        service = get_weather_service()
        return service.get_live_weather(location)
    except Exception as e:
        logger.error(f"Error in /api/weather: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch live weather: {str(e)}")


@router.get("/mandi-prices", tags=["Real-Time Data"])
async def get_live_mandi_prices():
    """
    Fetch real-time Indian APMC Mandi prices with Government of India Agmarknet integration,
    today's arrival dates, modal prices, price trends, and MSP 2024-25.
    """
    try:
        from services.mandi_service import get_mandi_service
        service = get_mandi_service()
        return service.get_realtime_mandi_data()
    except Exception as e:
        logger.error(f"Error in /api/mandi-prices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch live mandi prices: {str(e)}")


@router.get("/climate-risk", tags=["AI Climate Risk"])
async def get_crop_climate_risk(
    location: str = "Delhi",
    crop: str = "wheat",
    stage: Optional[str] = None,
):
    """
    AI Climate Risk Predictor & Early Warning System.
    Predicts drought risk, flood risk, heat stress, frost warnings,
    crop lodging, and fungal disease risk using satellite root zone
    soil moisture, past weather history, and 7-day meteorological forecasts.
    """
    try:
        from services.climate_risk_service import get_climate_risk_service
        service = get_climate_risk_service()
        return service.assess_crop_climate_risk(location=location, crop=crop, stage=stage)
    except Exception as e:
        logger.error(f"Error in /api/climate-risk: {e}")
        raise HTTPException(status_code=500, detail=f"Climate risk assessment failed: {str(e)}")


# ── Disease Detection ─────────────────────────────────────────────────────────

@router.post("/detect-disease", response_model=DiseaseDetectionResponse, tags=["Disease Detection"])
async def detect_disease(
    file: UploadFile = File(..., description="Plant/leaf image (JPEG or PNG)"),
    top_k: int = 3,
) -> DiseaseDetectionResponse:
    """
    Upload a plant or leaf image to detect diseases using YOLOv8.

    Accepts JPEG or PNG images. Returns the top-K disease detections
    with confidence scores, treatment advice, and prevention tips.

    - **file**: Image file (JPEG/PNG, max 10MB recommended)
    - **top_k**: Number of top results to return (default 3)
    """
    # Validate content type
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/webp", "application/pdf"}
    if file.content_type and file.content_type not in allowed_types and not (file.filename and file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".pdf"))):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload a JPEG, PNG, or PDF file.",
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # Max 20MB guard
        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large. Maximum size is 20MB.")

        # Extract image if PDF was uploaded
        is_pdf = file.content_type == "application/pdf" or (file.filename and file.filename.lower().endswith(".pdf"))
        if is_pdf:
            try:
                import io
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(image_bytes)
                if len(pdf) > 0:
                    page = pdf.get_page(0)
                    rendered = page.render(scale=2).to_pil()
                    buf = io.BytesIO()
                    rendered.convert("RGB").save(buf, format="JPEG", quality=95)
                    image_bytes = buf.getvalue()
            except Exception as e:
                logger.warning(f"PDF extraction error in /api/detect-disease: {e}")

        logger.info(f"Disease detection request — file={file.filename} size={len(image_bytes)} bytes top_k={top_k}")

        # Run YOLOv8 inference
        from disease_detector import get_detector
        detector = get_detector()
        raw_result = detector.detect(image_bytes, top_k=top_k)

        # Map raw dict to Pydantic model
        detections = [
            DetectionResult(**d) for d in raw_result.get("detections", [])
        ]
        primary_raw = raw_result.get("primary_disease")
        primary = DetectionResult(**primary_raw) if primary_raw else None

        return DiseaseDetectionResponse(
            success=raw_result["success"],
            detections=detections,
            primary_disease=primary,
            treatment=raw_result.get("treatment"),
            general_advice=raw_result.get("general_advice", {}),
            inference_time_ms=raw_result.get("inference_time_ms", 0.0),
            model=raw_result.get("model", ""),
            message=raw_result.get("message", ""),
            error=raw_result.get("error"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in /api/detect-disease: {e}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@router.get("/diseases", tags=["Disease Detection"])
async def list_diseases():
    """Return all supported disease classes with metadata."""
    import json
    from pathlib import Path

    data_path = Path(__file__).parent.parent.parent / "data" / "disease_classes.json"
    try:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        diseases = []
        for key, info in data.get("diseases", {}).items():
            diseases.append({
                "key": key,
                "common_name": info.get("common_name", key),
                "hindi_name": info.get("hindi_name", ""),
                "crop": info.get("crop", ""),
                "severity": info.get("severity", "unknown"),
            })
        return {
            "total": len(diseases),
            "diseases": diseases,
            "general_advice": data.get("general_advice", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Voice Assistant Endpoints ───────────────────────────────────────────────

@router.post("/voice/transcribe", tags=["Voice Assistant"])
async def voice_transcribe(
    file: UploadFile = File(..., description="Audio recording of farmer query (WAV, MP3, WebM, M4A)"),
    language: str = "auto",
):
    """Transcribe farmer's voice query using Whisper."""
    try:
        from voice_assistant import transcribe_audio
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file.")

        transcription = transcribe_audio(
            audio_bytes,
            filename=file.filename or "voice.wav",
            language=None if language == "auto" else language,
        )
        return {"success": True, "text": transcription, "language": language}
    except Exception as e:
        logger.error(f"Voice transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/voice/speak", tags=["Voice Assistant"])
async def voice_speak(
    text: str,
    language: str = "Hindi",
):
    """Convert agricultural advice text into natural speech (MP3)."""
    try:
        from fastapi.responses import Response
        from voice_assistant import generate_speech
        audio_bytes = generate_speech(text, language=language)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@router.post("/voice/assistant", tags=["Voice Assistant"])
async def voice_assistant_pipeline(
    file: UploadFile = File(..., description="Spoken voice query"),
    language: str = "Hindi",
    location: str = "Delhi",
):
    """
    End-to-end Multilingual Voice Farming Assistant:
    Speech-to-Text (Whisper) -> Farming Agent (RAG + Domain Tools) -> Text-to-Speech (gTTS).
    """
    try:
        from voice_assistant import transcribe_audio
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file.")

        # 1. Speech-to-Text
        query_text = transcribe_audio(
            audio_bytes,
            filename=file.filename or "query.wav",
            language=language,
        )
        if not query_text:
            query_text = "फसल सलाह"

        # 2. Run agent reasoning
        result = run_agent(
            message=query_text,
            language=language,
            location=location,
        )

        return {
            "success": True,
            "transcription": query_text,
            "response": result["response"],
            "sources": result.get("sources", []),
            "agent_steps": result.get("agent_steps", []),
            "language": language,
            "location": location,
        }
    except Exception as e:
        logger.error(f"Voice assistant pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
