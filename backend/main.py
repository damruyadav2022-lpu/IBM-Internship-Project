"""
KrishiRakshak - Smart Farming Advice AI Agent
FastAPI application entry point
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes import router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _background_startup():
    """Run heavy initialisation in a background thread so startup is non-blocking."""
    # 1. Ingest data into ChromaDB
    try:
        from utils.data_loader import DataLoader
        DataLoader().load_all()
        logger.info("Data ingestion into ChromaDB complete.")
    except Exception as e:
        logger.warning(f"Data ingestion warning (non-fatal): {e}")

    # 2. Pre-load YOLOv8 model
    try:
        from disease_detector import get_detector
        detector = get_detector()
        if detector.is_ready:
            logger.info("YOLOv8 disease detector loaded and ready.")
        else:
            logger.warning("YOLOv8 detector not ready — check YOLO_MODEL_PATH.")
    except Exception as e:
        logger.warning(f"YOLOv8 detector init warning (non-fatal): {e}")

    logger.info("Background startup complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — server starts immediately, heavy tasks run in background."""
    logger.info("Starting KrishiRakshak backend...")
    Path("uploads").mkdir(exist_ok=True)

    # Run heavy startup (embedding model, ChromaDB, YOLO) in a thread — non-blocking
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _background_startup)

    logger.info("KrishiRakshak backend ready — heavy init running in background.")
    yield
    logger.info("Shutting down KrishiRakshak backend.")


app = FastAPI(
    title="KrishiRakshak API",
    description="Smart Farming Advice AI Agent powered by RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the React frontend and any local dev origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router, prefix="/api")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to KrishiRakshak - Smart Farming Advice",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "KrishiRakshak Backend"}
