from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Union, Optional
import logging

from brand_brain.core.orchestrator import chat_session

router = APIRouter()

# --- Schemas ---

class AskRequest(BaseModel):
    brand_id: str
    question: str

class Source(BaseModel):
    asset_type: str
    source: str

class AskResponse(BaseModel):
    answer: str
    confidence: Union[float, str]
    intent: str
    sources: List[Source]

class Brand(BaseModel):
    id: str
    name: str

class BrandResponse(BaseModel):
    brands: List[Brand]

# --- Routes ---

@router.get("/brands", response_model=BrandResponse)
def get_supported_brands():
    """
    Returns static list of supported brands.
    No core dependency.
    """
    return BrandResponse(
        brands=[
            Brand(id="wh_india_001", name="Westinghouse India")
        ]
    )

@router.post("/ask", response_model=AskResponse)
def ask_brand_brain(request: AskRequest):
    try:
        # 1. Invoke Brand Brain Orchestrator
        # chat_session handles retrieval, reasoning, and validation
        response = chat_session(request.question, request.brand_id)
        
        # 2. Map Response (Strict Parity)
        # We strictly pass through values from the core.
        
        return AskResponse(
            answer=response.get("answer", ""),
            confidence=response.get("confidence_level", 0.0), # Pass through exact value
            intent=response.get("intent", "UNKNOWN"),         # Pass through exact value
            sources=[] # Core limitation: returns no source metadata
        )

    except Exception as e:
        # Log error safely
        logging.error(f"Error processing request: {e}", exc_info=False)
        
        # Classify Error for Status Code
        error_msg = str(e).lower()
        if "pinecone" in error_msg or "gemini" in error_msg or "service unavailable" in error_msg:
             raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service currently unavailable")
        
        # Default to 500 for internal logic/DB errors
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error")
