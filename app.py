"""
FastAPI application for passport scanning
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List
from config import config
from scanner import scan_passport
from function_handler_switch import (
    get_step_status, 
    enable_step, 
    disable_step, 
    save_step_config
)


# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class ScanRequest(BaseModel):
    """Request model for document scanning - supports both images and PDF files"""
    image_type: str = Field(..., description="Type of input: 'file' for URL, 'base64' for base64 encoded (supports images and PDFs)")
    documents_image_url: Optional[str] = Field(None, description="URL of the document image or PDF (required if image_type='file')")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image or PDF data (required if image_type='base64'). PDFs will be auto-converted to images.")
    documents_type: str = Field(default="passport", description="Type of document: passport, id_card, visa")
    step_config: Optional[Dict[str, bool]] = Field(None, description="Optional step configuration override for this scan (e.g., {'STEP1': true, 'STEP2': false, ...})")
    
    @field_validator('image_type')
    @classmethod
    def validate_image_type(cls, v):
        if v not in ['file', 'base64']:
            raise ValueError('image_type must be either "file" or "base64"')
        return v
    
    @field_validator('documents_type')
    @classmethod
    def validate_document_type(cls, v):
        if v not in config.SUPPORTED_DOCUMENT_TYPES:
            raise ValueError(f'documents_type must be one of {config.SUPPORTED_DOCUMENT_TYPES}')
        return v
    
    def model_post_init(self, __context):
        """Validate that the appropriate image source is provided based on image_type"""
        if self.image_type == 'file' and not self.documents_image_url:
            raise ValueError('documents_image_url is required when image_type is "file"')
        if self.image_type == 'base64' and not self.image_base64:
            raise ValueError('image_base64 is required when image_type is "base64"')


# Response models
class PassportData(BaseModel):
    """Passport data model"""
    document_type: str = ""
    country_code: str = ""
    surname: str = ""
    given_names: str = ""
    passport_number: str = ""
    country_name: str = ""
    country_iso: str = ""
    nationality: str = ""
    date_of_birth: str = ""
    sex: str = ""
    place_of_Birth: Optional[str] = ""
    issue_date: Optional[str] = ""
    expiry_date: str = ""
    personal_number: str = ""


class ScanResponse(BaseModel):
    """Response model for document scanning"""
    success: bool
    passport_data: Dict
    mrz_text: str = ""
    working_process_step: Dict = {}
    step_timings: Dict = {}
    total_time: str = ""
    error: str = ""
    validation_reason: Optional[str] = ""
    validation_confidence: Optional[str] = ""
    
    class Config:
        # Allow extra fields in the response
        extra = "allow"


class StepConfigRequest(BaseModel):
    """Request model for step configuration"""
    steps: Dict[str, bool] = Field(..., description="Dictionary of step names and their enabled status")


class StepConfigResponse(BaseModel):
    """Response model for step configuration"""
    success: bool
    steps: Dict[str, bool]
    message: str = ""


# API endpoints
@app.get("/")
async def root():
    """Root endpoint - serves the HTML UI"""
    import os
    # Get the directory where app.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "index.html")
    
    # Check if file exists
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        # Fallback to JSON response if HTML not found
        return {
            "message": "Passport Scanner API",
            "version": config.API_VERSION,
            "note": "index.html not found. Please ensure index.html is in the same directory as app.py",
            "endpoints": {
                "scan": "/scan - POST endpoint to scan passport/ID card images",
                "health": "/health - GET endpoint to check API health",
                "docs": "/docs - Swagger UI documentation"
            }
        }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Passport Scanner API",
        "version": config.API_VERSION
    }


@app.post("/scan", response_model=ScanResponse)
async def scan_document(
    request: ScanRequest,
    ai: str = "on"  # Query parameter: ai=on or ai=off, default 'on'
):
    """
    Scan passport or ID card image/PDF and extract data
    
    Supports both image files (JPG, PNG, etc.) and PDF files.
    PDFs are automatically converted to images (first page is used).
    
    Args:
        request: ScanRequest object with image/PDF URL or base64 data
        ai: Control AI usage ('on' or 'off'), default 'on'
        
    Returns:
        ScanResponse with extracted passport data
        
    Examples:
        POST /scan (uses AI by default)
        POST /scan?ai=on (explicitly use AI)
        POST /scan?ai=off (use traditional OCR)
        
        With image URL:
        ```json
        {
            "image_type": "file",
            "documents_image_url": "https://example.com/passport.jpg",
            "documents_type": "passport"
        }
        ```
        
        With base64 PDF:
        ```json
        {
            "image_type": "base64",
            "image_base64": "JVBERi0xLjQKJeLjz9...",
            "documents_type": "passport"
        }
        ```
    """
    try:
        # Determine if Gemini should be used (default is True)
        use_gemini = ai.lower() != "off"  # Use AI unless explicitly set to 'off'
        
        # Scan the document based on image_type
        if request.image_type == "file":
            # URL-based image
            result = scan_passport(
                image_url=request.documents_image_url,
                document_type=request.documents_type,
                use_gemini=use_gemini,
                step_config_override=request.step_config
            )
        elif request.image_type == "base64":
            # Base64-encoded image
            result = scan_passport(
                image_base64=request.image_base64,
                document_type=request.documents_type,
                use_gemini=use_gemini,
                step_config_override=request.step_config
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image_type: {request.image_type}"
            )
        
        # If extraction failed, return 422 status code with full response
        if not result.get("success"):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=422,
                content=result
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@app.get("/step-config", response_model=StepConfigResponse)
async def get_step_config():
    """
    Get current step configuration
    
    Returns:
        StepConfigResponse with current step status
    """
    try:
        steps = get_step_status()
        return StepConfigResponse(
            success=True,
            steps=steps,
            message="Step configuration retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving step configuration: {str(e)}"
        )


@app.post("/step-config", response_model=StepConfigResponse)
async def update_step_config(request: StepConfigRequest):
    """
    Update step configuration
    
    Args:
        request: StepConfigRequest with new step configuration
        
    Returns:
        StepConfigResponse with updated step status
        
    Example:
        ```json
        {
            "steps": {
                "STEP1": true,
                "STEP2": true,
                "STEP3": false,
                "STEP4": false,
                "STEP5": true,
                "STEP6": false
            }
        }
        ```
    """
    try:
        # Update each step based on the request
        for step_name, enabled in request.steps.items():
            step_name = step_name.upper()
            if step_name in ["STEP1", "STEP2", "STEP3", "STEP4", "STEP5", "STEP6"]:
                if enabled:
                    enable_step(step_name)
                else:
                    disable_step(step_name)
        
        # Save configuration to file
        save_step_config()
        
        # Return updated configuration
        updated_steps = get_step_status()
        return StepConfigResponse(
            success=True,
            steps=updated_steps,
            message="Step configuration updated successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating step configuration: {str(e)}"
        )


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
