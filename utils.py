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


def save_passport_page_crop(image: Image.Image, user_folder: Path = None) -> str:
    """
    Save passport page crop as passport_page_crop.jpg
    
    Args:
        image: PIL Image object of the cropped passport page
        user_folder: Optional user-specific folder path
        
    Returns:
        Path to saved passport_page_crop.jpg file
    """
    try:
        # Determine save path
        if user_folder:
            user_path = Path(user_folder)
            user_path.mkdir(parents=True, exist_ok=True)
            save_path = user_path / 'passport_page_crop.jpg'
        else:
            temp_dir = Path('temp')
            temp_dir.mkdir(exist_ok=True)
            save_path = temp_dir / 'passport_page_crop.jpg'
        
        # Convert RGBA to RGB if needed (JPEG doesn't support alpha channel)
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = rgb_image
        
        # Save the image
        image.save(str(save_path), 'JPEG', quality=95)
        print(f"Saved passport page crop: {save_path}")
        
        return str(save_path)
        
    except Exception as e:
        print(f"⚠️ Failed to save passport page crop: {e}")
        return ""


def cleanup_user_folder(user_folder: Path):
    """
    Delete all files in a user's temporary folder and associated validation failure files
    
    Args:
        user_folder: Path to user's temp folder
    """
    try:
        # DEBUG MODE: Don't delete files for debugging purposes
        print(f"  DEBUG MODE: Skipping cleanup of user folder: {user_folder}")
        print(f"  Files preserved for debugging in: {user_folder}")
        
        # In production, you would uncomment the following:
        # if user_folder and user_folder.exists():
        #     # Extract user_id from folder name to find associated validation files
        #     user_id = user_folder.name
        #     
        #     # Delete all files in the folder
        #     for file in user_folder.glob("*"):
        #         if file.is_file():
        #             file.unlink()
        #     # Delete the folder itself
        #     user_folder.rmdir()
        #     print(f"  ✓ Cleaned up user folder: {user_folder.name}")
        #     
        #     # Also delete associated validation failure JSON files
        #     temp_dir = user_folder.parent
        #     validation_files = list(temp_dir.glob(f"validation_failures_{user_id}.json"))
        #     for validation_file in validation_files:
        #         if validation_file.exists():
        #             validation_file.unlink()
        #             print(f"  ✓ Cleaned up validation file: {validation_file.name}")
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


def check_field_validation_threshold(mrz_text: str, threshold: int = 10, verbose: bool = False) -> dict:
    """
    Check if MRZ field validation meets the required threshold
    
    Args:
        mrz_text: MRZ text to validate
        threshold: Minimum number of valid fields required (default: 10)
        verbose: Print validation details
        
    Returns:
        Dictionary with validation results:
        {
            "threshold_met": bool,
            "valid_count": int,
            "total_count": int,
            "field_results": dict,
            "summary": str
        }
    """
    try:
        from passport_check import validate_passport_fields
        
        # Validate all fields
        field_results = validate_passport_fields(mrz_text)
        
        # Count valid fields
        valid_count = sum(1 for status in field_results.values() if status == "Valid")
        total_count = len(field_results)
        
        # Check threshold
        threshold_met = valid_count >= threshold
        summary = f"{valid_count}/{total_count} fields are valid"
        
        if verbose:
            print(f"\n🔍 FIELD VALIDATION CHECK:")
            for field, status in field_results.items():
                status_icon = "✅" if status == "Valid" else "❌"
                print(f"  {status_icon} {field:20}: {status}")
            
            print(f"\n📊 Validation Summary: {summary}")
            print(f"   Threshold: {threshold}/10 fields required")
            print(f"   Result: {'✅ PASSED' if threshold_met else '❌ FAILED'}")
        
        return {
            "threshold_met": threshold_met,
            "valid_count": valid_count,
            "total_count": total_count,
            "field_results": field_results,
            "summary": summary
        }
        
    except Exception as e:
        if verbose:
            print(f"❌ Error during field validation: {e}")
        
        return {
            "threshold_met": False,
            "valid_count": 0,
            "total_count": 10,
            "field_results": {},
            "summary": "Validation error"
        }


def save_validation_failure(user_id: str, method_name: str, passport_data: dict, field_results: dict, mrz_text: str = "", full_text_preview: str = "") -> str:
    """
    Save validation failure data to a temporary JSON file for the user
    
    Args:
        user_id: Unique user identifier
        method_name: OCR method name (FastMRZ, PassportEye, EasyOCR, Tesseract)
        passport_data: Original passport data extracted
        field_results: Field validation results from passport_check
        mrz_text: MRZ text for reference
        
    Returns:
        Path to the validation failure file
    """
    import json
    import os
    from pathlib import Path
    
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Create validation failure filename
        validation_file = temp_dir / f"validation_failures_{user_id}.json"
        
        # Load existing data if file exists
        validation_data = {}
        if validation_file.exists():
            try:
                with open(validation_file, 'r', encoding='utf-8') as f:
                    validation_data = json.load(f)
            except Exception:
                validation_data = {}
        
        # Extract field errors (only invalid fields)
        field_errors = {}
        for field, status in field_results.items():
            if status == "Invalid":
                # Determine error type based on field and data
                if field == "date_of_birth":
                    birth_date = passport_data.get("date_of_birth", "")
                    if "00" in birth_date or len(birth_date.replace("-", "")) != 8:
                        field_errors[field] = "invalid_format"
                    else:
                        field_errors[field] = "invalid_date"
                elif field == "expiry_date":
                    expiry_date = passport_data.get("expiry_date", "")
                    if "00" in expiry_date or len(expiry_date.replace("-", "")) != 8:
                        field_errors[field] = "invalid_format"
                    else:
                        field_errors[field] = "invalid_date"
                elif field == "sex":
                    sex_value = passport_data.get("sex", "")
                    if sex_value not in ["M", "F", "X"]:
                        field_errors[field] = "invalid_value"
                elif field in ["document_type", "issuing_country", "nationality"]:
                    field_errors[field] = "invalid_code"
                elif field in ["surname", "given_names"]:
                    field_errors[field] = "invalid_format"
                elif field in ["passport_number", "personal_number"]:
                    field_errors[field] = "invalid_format"
                else:
                    field_errors[field] = "validation_failed"
        
        # Add method data to validation file
        method_data = {
            "original_data": passport_data.copy(),
            "field_errors": field_errors,
            "mrz_text": mrz_text,
            "timestamp": datetime.now().isoformat(),
            "valid_count": sum(1 for status in field_results.values() if status == "Valid"),
            "total_count": len(field_results)
        }
        
        # Add full text preview if provided (especially useful for EasyOCR)
        if full_text_preview:
            method_data["full_text_preview"] = full_text_preview
        
        validation_data[method_name] = method_data
        
        # Save updated validation data
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved validation failure for {method_name} to: {validation_file}")
        print(f"   → Field errors: {list(field_errors.keys())}")
        
        return str(validation_file)
        
    except Exception as e:
        print(f"⚠️ Error saving validation failure: {e}")
        return ""


def load_validation_failures(user_id: str) -> dict:
    """
    Load validation failure data for a user
    
    Args:
        user_id: Unique user identifier
        
    Returns:
        Dictionary with validation failure data from previous methods
    """
    import json
    from pathlib import Path
    
    try:
        validation_file = Path("temp") / f"validation_failures_{user_id}.json"
        
        if validation_file.exists():
            with open(validation_file, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
            
            print(f"📂 Loaded validation failures from: {validation_file}")
            print(f"   → Previous methods: {list(validation_data.keys())}")
            
            return validation_data
        else:
            print(f"📂 No validation failure file found for user: {user_id}")
            return {}
            
    except Exception as e:
        print(f"⚠️ Error loading validation failures: {e}")
        return {}


def remove_validation_failures(user_id: str):
    """
    Remove validation failure file when processing is successful
    
    Args:
        user_id: Unique user identifier
    """
    from pathlib import Path
    
    try:
        validation_file = Path("temp") / f"validation_failures_{user_id}.json"
        
        if validation_file.exists():
            validation_file.unlink()
            print(f"🗑️ Removed validation failure file: {validation_file}")
        
    except Exception as e:
        print(f"⚠️ Error removing validation failure file: {e}")


def analyze_previous_failures(validation_data: dict, current_method: str) -> dict:
    """
    Analyze previous validation failures to provide insights for current method
    
    Args:
        validation_data: Previous validation failure data
        current_method: Current OCR method name
        
    Returns:
        Dictionary with analysis and suggestions
    """
    if not validation_data:
        return {"suggestions": [], "common_errors": [], "previous_methods": []}
    
    # Collect all field errors from previous methods
    all_errors = {}
    previous_methods = list(validation_data.keys())
    
    for method, data in validation_data.items():
        field_errors = data.get("field_errors", {})
        for field, error_type in field_errors.items():
            if field not in all_errors:
                all_errors[field] = []
            all_errors[field].append(f"{method}: {error_type}")
    
    # Find most common errors
    common_errors = []
    for field, errors in all_errors.items():
        if len(errors) > 1:  # Error occurred in multiple methods
            common_errors.append(f"{field} ({len(errors)} methods)")
    
    # Generate suggestions based on common patterns
    suggestions = []
    if "date_of_birth" in all_errors:
        suggestions.append("Focus on birth date extraction - common issue across methods")
    if "sex" in all_errors:
        suggestions.append("Sex field validation failing - check for non-standard values")
    if "expiry_date" in all_errors:
        suggestions.append("Expiry date format issues detected")
    
    analysis = {
        "suggestions": suggestions,
        "common_errors": common_errors,
        "previous_methods": previous_methods,
        "total_previous_attempts": len(previous_methods)
    }
    
    print(f"\n🔍 ANALYSIS OF PREVIOUS FAILURES:")
    print(f"   → Previous methods tried: {', '.join(previous_methods)}")
    print(f"   → Common error fields: {', '.join(common_errors) if common_errors else 'None'}")
    print(f"   → Suggestions: {len(suggestions)} recommendations")
    
    return analysis


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
        
        # Clean up old validation failure files (older than 2 hours)
        for file in config.TEMP_DIR.glob("validation_failures_*.json"):
            if (datetime.now().timestamp() - file.stat().st_mtime) > 7200:  # 2 hours
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
