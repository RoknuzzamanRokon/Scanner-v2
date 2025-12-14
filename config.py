"""
Configuration settings for Passport Scanner API
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""
    
    # API Settings
    API_TITLE = "Passport Scanner API"
    API_VERSION = "2.0.0"
    API_DESCRIPTION = """
    Advanced passport and ID card scanning API with multi-layered fallback system.
    
    Features:
    - PassportEye MRZ detection
    - FastMRZ library integration
    - AI-powered extraction using Gemini
    - TD3 format validation
    - Multiple extraction methods with fallback
    """
    
    # Document Types
    SUPPORTED_DOCUMENT_TYPES = ["passport", "id_card", "visa"]
    
    # Temp Directory (as Path object for proper path operations)
    TEMP_DIR = Path(__file__).parent / "temp"
    
    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Tesseract OCR
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
    
    # Processing Settings
    DEFAULT_USE_GEMINI = True
    MIN_CONFIDENCE_THRESHOLD = 0.5  # 50% confidence required
    
    # Image Processing
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    DOWNLOAD_TIMEOUT = 30  # seconds timeout for image download
    SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp"]
    SUPPORTED_PDF_EXTENSIONS = ["pdf"]
    
    # MRZ Settings
    TD3_LINE_LENGTH = 44
    TD3_TOTAL_LINES = 2
    
    # Validation
    VALID_SEX_VALUES = ["M", "F", "X", "O", "NB", "<"]
    
    @classmethod
    def ensure_temp_dir(cls):
        """Ensure temp directory exists"""
        os.makedirs(cls.TEMP_DIR, exist_ok=True)


# Create global config instance
config = Config()

# Ensure temp directory exists on import
config.ensure_temp_dir()
