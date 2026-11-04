"""
Pydantic request/response models for KrishiRakshak API.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    """Request body for the /api/chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Farmer's question or message",
        examples=["What crop should I grow this Kharif season?"],
    )
    language: str = Field(
        default="English",
        description="Response language: 'English' or 'Hindi'",
        examples=["English", "Hindi"],
    )
    location: str = Field(
        default="Delhi",
        description="Farmer's location (city name) for localised advice",
        examples=["Delhi", "Lucknow", "Pune", "Jaipur", "Mumbai"],
    )


class AgentStep(BaseModel):
    """A single step taken by the agent during reasoning."""

    tool: str = Field(..., description="Tool name used")
    input: str = Field(..., description="Input provided to the tool")
    output: str = Field(..., description="Output from the tool (truncated)")


class ChatResponse(BaseModel):
    """Response body for the /api/chat endpoint."""

    response: str = Field(..., description="AI agent's answer to the farmer's question")
    sources: List[str] = Field(
        default_factory=list,
        description="List of tools/data sources consulted",
    )
    agent_steps: List[AgentStep] = Field(
        default_factory=list,
        description="Intermediate reasoning steps taken by the agent",
    )
    language: str = Field(default="English", description="Response language used")
    location: str = Field(default="Delhi", description="Location context used")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str = "1.0.0"


# ── Disease Detection Models ──────────────────────────────────────────────────

class DetectionResult(BaseModel):
    """A single disease detection result from YOLOv8."""

    class_id: int = Field(..., description="YOLOv8 class index")
    class_name: str = Field(..., description="Raw class name from model")
    confidence: float = Field(..., description="Detection confidence (0–1)")
    display_name: str = Field(..., description="Human-readable disease name")
    hindi_name: str = Field(default="", description="Disease name in Hindi")
    severity: str = Field(
        default="unknown",
        description="Disease severity: none | medium | high | critical | unknown",
    )
    bbox: Optional[Dict[str, float]] = Field(
        default=None,
        description="Bounding box {x1, y1, x2, y2} in pixels (detection models only)",
    )


class TreatmentInfo(BaseModel):
    """Treatment and prevention information for a detected disease."""

    common_name: str = ""
    hindi_name: str = ""
    crop: str = ""
    severity: str = ""
    symptoms: str = ""
    cause: str = ""
    favorable_conditions: str = ""
    organic_treatment: str = ""
    chemical_treatment: str = ""
    prevention: str = ""
    economic_impact: str = ""


class DiseaseDetectionResponse(BaseModel):
    """Response body for POST /api/detect-disease."""

    success: bool = Field(..., description="Whether inference succeeded")
    detections: List[DetectionResult] = Field(
        default_factory=list,
        description="Top-K disease detections sorted by confidence",
    )
    primary_disease: Optional[DetectionResult] = Field(
        default=None,
        description="Highest-confidence detection",
    )
    treatment: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Treatment & prevention info for primary disease",
    )
    general_advice: Dict[str, Any] = Field(
        default_factory=dict,
        description="General IPM and safety advice",
    )
    inference_time_ms: float = Field(
        default=0.0,
        description="Model inference time in milliseconds",
    )
    model: str = Field(default="", description="Model filename used")
    message: str = Field(default="", description="Human-readable summary")
    error: Optional[str] = Field(default=None, description="Error message if failed")
