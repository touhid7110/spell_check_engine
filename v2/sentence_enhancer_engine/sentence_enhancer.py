from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Dict, Any
import json
import logging
from enhancement_service import EnhancementService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class StyleType(str, Enum):
    FORMAL = "formal"
    CONCISE = "concise"
    ELABORATE = "elaborate"
    SIMPLER = "simpler"
    BATMAN = "batman"

class EnhancementRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Text to enhance")
    style: StyleType = Field(..., description="Enhancement style to apply")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "I have merely adopted the dark",
                "style": "formal"
            }
        }

class EnhancementResponse(BaseModel):
    enhanced_reccomendations: Dict[str, Any] = Field(..., description="Enhanced text recommendations")
    style: str = Field(..., description="Applied enhancement style")
    original_text: str = Field(..., description="Original input text")
    success: bool = Field(default=True, description="Request success status")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: str = Field(None, description="Additional error details")
    success: bool = Field(default=False, description="Request success status")

# FastAPI App Instance
enhancer_app = FastAPI(
    title="Sentence Enhancement API",
    description="AI-powered sentence rephrasing and style enhancement service using Google's Gemma model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
enhancer_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Enhancement Service
try:
    enhancement_service = EnhancementService()
    logger.info("Enhancement service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize enhancement service: {str(e)}")
    enhancement_service = None

# Health Check Endpoint
@enhancer_app.get("/health")
def health_check():
    """Health check endpoint to verify API is running"""
    return {
        "status": "healthy",
        "service": "Sentence Enhancement API",
        "version": "1.0.0",
        "enhancement_service_available": enhancement_service is not None
    }

# Main Enhancement Endpoint
@enhancer_app.post(
    "/api/v1/enhance",
    response_model=EnhancementResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
def enhance_sentence(request: EnhancementRequest):
    """
    Enhance a sentence with the specified style using AI
    
    - **text**: The sentence to enhance (1-1000 characters)
    - **style**: Enhancement style (formal, concise, elaborate, simpler, batman)
    
    Returns enhanced text suggestions in JSON format
    """
    
    if not enhancement_service:
        logger.error("Enhancement service not available")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Enhancement service is not available"
        )
    
    try:
        # Log request
        logger.info(f"Enhancement request - Style: {request.style}, Text length: {len(request.text)}")
        
        # Call enhancement service
        result = enhancement_service.enhance(
            text=request.text.strip(),
            style=request.style.value
        )
        
        # Log successful response
        logger.info(f"Enhancement successful for style: {request.style}")
        
        return EnhancementResponse(
            enhanced_reccomendations=result["enhanced_reccomendations"],
            style=result["style"],
            original_text=request.text,
            success=True
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse AI response: {str(e)}"
        )
    
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during enhancement: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhancement failed: {str(e)}"
        )

# List Available Styles Endpoint
@enhancer_app.get("/api/v1/styles")
def get_available_styles():
    """Get list of available enhancement styles"""
    try:
        if not enhancement_service:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Enhancement service not available"
            )
        
        styles = enhancement_service.prompt_manager.get_available_styles()
        return {
            "available_styles": styles,
            "total_count": len(styles)
        }
    except Exception as e:
        logger.error(f"Error retrieving styles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve available styles"
        )

# Root endpoint
@enhancer_app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Sentence Enhancement API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "enhancement_endpoint": "/api/v1/enhance"
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "sentence_enhancer:enhancer_app",
        host="0.0.0.0",
        port=7110,
        reload=True,
        log_level="info"
    )
