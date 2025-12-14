"""
Utility functions for image processing and data formatting
"""
import io
import requests
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from config import config


def download_image(url: str) -> Image.Image:
    """
    Download image from URL and return as PIL Image
    
    Args:
        url: Image URL
        
    Returns:
        PIL Image object
        
    Raises:
        Exception: If download fails or image is invalid
    """
    try:
        response = requests.get(url, timeout=config.DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
        
        # Check file size
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > config.MAX_IMAGE_SIZE:
            raise Exception(f"Image too large: {content_length} bytes")
        
        # Open image
        image = Image.open(io.BytesIO(response.content))
        return image
        
    except requests.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to process image: {str(e)}")


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string and return as PIL Image
    Supports both image files and PDF files (converts first page)
    
    Args:
        base64_string: Base64 encoded image or PDF string
        
    Returns:
        PIL Image object
        
    Raises:
        Exception: If decoding fails or file is invalid
    """
    import base64
    from pdf_utils import is_pdf, get_first_page_as_image
    
    try:
        # Store original input for PDF detection
        original_input = base64_string
        
        # Check if input is a PDF
        if is_pdf(original_input):
            print("→ PDF file detected, converting to image...")
            return get_first_page_as_image(original_input, dpi=300)
        
        # Otherwise, process as regular image
        # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_string and base64_string.startswith('data:'):
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode base64 string
        image_bytes = base64.b64decode(base64_string)
        
        # Check size limit
        if len(image_bytes) > config.MAX_IMAGE_SIZE:
            raise Exception(f"Image too large: {len(image_bytes)} bytes")
        
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        return image
        
    except Exception as e:
        raise Exception(f"Failed to decode base64 data: {str(e)}")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for better OCR results
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed image as numpy array
    """
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def preprocess_for_mrz(image: Image.Image) -> Image.Image:
    """
    Preprocess image specifically for MRZ extraction
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed PIL Image
    """
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(2.0)
    
    # Enhance sharpness
    sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
    sharpened = sharpness_enhancer.enhance(2.0)
    
    return sharpened


def format_mrz_date(mrz_date: str) -> str:
    """
    Convert MRZ date format (YYMMDD) to readable format (YYYY-MM-DD)
    
    Args:
        mrz_date: Date in YYMMDD format
        
    Returns:
        Date in YYYY-MM-DD format
    """
    if not mrz_date or len(mrz_date) != 6:
        return ""
    
    try:
        from datetime import datetime
        
        year = int(mrz_date[0:2])
        month = mrz_date[2:4]
        day = mrz_date[4:6]
        
        current_year = datetime.now().year
        current_year_2digit = current_year % 100
        
        # Smart century determination:
        # Try 2000s first
        full_year_2000s = 2000 + year
        
        # If the resulting year is more than 10 years in the future,
        # it's likely supposed to be in the 1900s (e.g., DOB)
        if full_year_2000s > current_year + 15:
            full_year = 1900 + year
        else:
            full_year = full_year_2000s
        
        return f"{full_year}-{month}-{day}"
    except:
        return ""


def format_issue_expiry_date(date_str: str) -> str:
    """
    Format issue/expiry date from OCR text
    
    Args:
        date_str: Date string from OCR
        
    Returns:
        Formatted date string
    """
    # This is a placeholder - actual implementation depends on the format found in OCR
    return date_str


def clean_text(text: str) -> str:
    """
    Clean OCR text by removing extra whitespace and invalid characters
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with single space
    cleaned = ' '.join(text.split())
    return cleaned


def validate_mrz_checksum(data: str, check_digit: str) -> bool:
    """
    Validate MRZ checksum digit
    
    Args:
        data: Data string to validate
        check_digit: Expected check digit
        
    Returns:
        True if checksum is valid
    """
    weights = [7, 3, 1]
    total = 0
    
    for i, char in enumerate(data):
        if char.isdigit():
            value = int(char)
        elif char == '<':
            value = 0
        else:
            # A-Z: A=10, B=11, ..., Z=35
            value = ord(char) - ord('A') + 10
        
        total += value * weights[i % 3]
    
    return str(total % 10) == check_digit


def create_user_temp_folder(user_id: str = None) -> Path:
    """
    Create a unique temporary folder for a user
    
    Args:
        user_id: Optional user ID. If not provided, generates a unique ID
        
    Returns:
        Path to user's temp folder
    """
    import uuid
    
    if not user_id:
        # Generate unique ID using timestamp + UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        user_id = f"{timestamp}_{unique_id}"
    
    # Create user-specific folder
    user_folder = config.TEMP_DIR / user_id
    user_folder.mkdir(parents=True, exist_ok=True)
    
    return user_folder


def get_user_id_from_url(image_url: str) -> str:
    """
    Generate a unique user ID from image URL
    
    Args:
        image_url: URL of the image
        
    Returns:
        Sanitized user ID based on URL
    """
    import hashlib
    import re
    
    # Extract meaningful parts from URL
    # Example: https://hoteljson.innsightmap.com/test1/1 (2).jpeg
    # Result: test1_1_2_jpeg
    
    # Remove protocol
    url_part = re.sub(r'^https?://', '', image_url)
    
    # Remove domain
    url_part = re.sub(r'^[^/]+/', '', url_part)
    
    # Replace special characters with underscores
    url_part = re.sub(r'[^\w\-.]', '_', url_part)
    
    # Remove multiple underscores
    url_part = re.sub(r'_+', '_', url_part)
    
    # Remove leading/trailing underscores
    url_part = url_part.strip('_')
    
    # Limit length and add hash for uniqueness
    if len(url_part) > 50:
        # Use hash of full URL + shortened path
        url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
        url_part = f"{url_part[:40]}_{url_hash}"
    
    return url_part if url_part else hashlib.md5(image_url.encode()).hexdigest()[:16]


def get_user_id_from_base64(base64_string: str) -> str:
    """
    Generate a unique user ID from base64 image data
    
    Args:
        base64_string: Base64 encoded image
        
    Returns:
        Unique user ID based on image hash
    """
    import hashlib
    import uuid
    from datetime import datetime
    
    # Create a hash from the base64 data (first 1000 chars for performance)
    data_sample = base64_string[:1000] if len(base64_string) > 1000 else base64_string
    data_hash = hashlib.md5(data_sample.encode()).hexdigest()[:8]
    
    # Add timestamp with microseconds for uniqueness (handles concurrent users)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    return f"base64_{timestamp}_{data_hash}"


def cleanup_user_folder(user_folder: Path):
    """
    Delete all files in a user's temporary folder
    
    Args:
        user_folder: Path to user's temp folder
    """
    try:
        if user_folder and user_folder.exists():
            # Delete all files in the folder
            for file in user_folder.glob("*"):
                if file.is_file():
                    file.unlink()
            # Delete the folder itself
            user_folder.rmdir()
            print(f"  ✓ Cleaned up user folder: {user_folder.name}")
        # print(f"  ✓ Cleaned up user folder")
    except Exception as e:
        print(f"  ⚠ Failed to cleanup user folder: {e}")


def save_temp_image(image: Image.Image, prefix: str = "temp", user_folder: Path = None) -> Path:
    """
    Save image to temporary directory (user-specific or global)
    
    Args:
        image: PIL Image
        prefix: Filename prefix
        user_folder: Optional user-specific folder path
        
    Returns:
        Path to saved image
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    
    # Use user folder if provided, otherwise use global temp dir
    if user_folder:
        filepath = user_folder / filename
    else:
        filepath = config.TEMP_DIR / filename
    
    # Convert RGBA to RGB if needed (JPEG doesn't support alpha channel)
    if image.mode in ('RGBA', 'LA', 'P'):
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
        image = rgb_image
        
    image.save(filepath, 'JPEG')
    return filepath


def cleanup_temp_files():
    """
    Clean up old temporary files and folders
    """
    try:
        # Clean up old individual files in root temp dir
        for file in config.TEMP_DIR.glob("*.jpg"):
            # Delete files older than 1 hour
            if (datetime.now().timestamp() - file.stat().st_mtime) > 3600:
                file.unlink()
        
        # Clean up old user folders (older than 1 hour)
        for folder in config.TEMP_DIR.iterdir():
            if folder.is_dir():
                # Check if folder is older than 1 hour
                if (datetime.now().timestamp() - folder.stat().st_mtime) > 3600:
                    # Delete all files in the folder
                    for file in folder.glob("*"):
                        if file.is_file():
                            file.unlink()
                    # Delete the folder
                    folder.rmdir()
    except Exception:
        pass  # Ignore cleanup errors
