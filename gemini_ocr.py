"""
Gemini AI OCR integration for passport data extraction
Enhanced version with better error handling, configuration, and performance
"""
import os
import google.generativeai as genai
from PIL import Image
from typing import Optional, Tuple
from config import config
import time
import re


# Configure Gemini API
if config.GEMINI_API_KEY:
    genai.configure(api_key=config.GEMINI_API_KEY)
else:
    print("⚠ Warning: GEMINI_API_KEY not found in environment variables")


class GeminiOCRError(Exception):
    """Custom exception for Gemini OCR errors"""
    def __init__(self, message: str, error_type: str = "general"):
        self.message = message
        self.error_type = error_type
        super().__init__(f"{error_type.upper()}: {message}")


def _validate_api_key() -> None:
    """Validate that Gemini API key is configured"""
    if not config.GEMINI_API_KEY:
        raise GeminiOCRError("GEMINI_API_KEY not configured", "configuration")


def _get_model() -> genai.GenerativeModel:
    """Get the configured Gemini model with fallback"""
    model_name = config.GEMINI_MODEL if hasattr(config, 'GEMINI_MODEL') and config.GEMINI_MODEL else 'gemini-2.0-flash'
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        # Fallback to default model if configured model fails
        if model_name != 'gemini-2.0-flash':
            print(f"⚠ Warning: Model '{model_name}' not available, falling back to 'gemini-2.0-flash'")
            return genai.GenerativeModel('gemini-2.0-flash')
        raise GeminiOCRError(f"Failed to initialize model: {str(e)}", "model_initialization")


def _clean_mrz_response(response_text: str) -> str:
    """Clean and validate MRZ response text"""
    if not response_text:
        return ""
    
    # Remove markdown code blocks
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        clean_lines = [line for line in lines if not line.startswith('```') and line.strip()]
        response_text = '\n'.join(clean_lines).strip()
    
    # Remove any non-MRZ text (lines that don't match MRZ pattern)
    mrz_lines = []
    for line in response_text.split('\n'):
        line = line.strip()
        # MRZ lines should be 44 characters and contain specific patterns
        if len(line) == 44 and ('<' in line or line.replace('<', '').isalnum()):
            mrz_lines.append(line)
    
    # Return first 2 lines if we have them
    if len(mrz_lines) >= 2:
        return f"{mrz_lines[0]}\n{mrz_lines[1]}"
    elif len(mrz_lines) == 1:
        return mrz_lines[0]
    else:
        return ""


def _retry_api_call(func, max_retries: int = 2, delay: float = 1.0) -> any:
    """Retry API call with exponential backoff for transient errors"""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_msg = str(e)
            
            # Don't retry for authentication/quota errors
            if "API_KEY" in error_msg or "api_key" in error_msg:
                break
            if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                break
            if "permission" in error_msg.lower():
                break
            
            # Retry for transient errors
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                print(f"⚠ API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
    
    if last_exception:
        raise last_exception


def extract_mrz_with_gemini(image: Image.Image, max_retries: int = 2) -> Optional[str]:
    """
    Extract MRZ text from passport image using Gemini Vision (Optimized approach)
    
    This is the PRIORITY method for AI extraction - faster and more focused
    
    Args:
        image: PIL Image object
        max_retries: Maximum number of API call retries (default: 2)
        
    Returns:
        MRZ text (2 lines) or empty string if extraction fails
        
    Raises:
        GeminiOCRError: If API call fails with specific error information
    """
    try:
        _validate_api_key()
        
        # Enhanced prompt with better structure and examples
        prompt = """Extract ONLY the Machine Readable Zone (MRZ) text from this passport image.

The MRZ is the 2-line text at the bottom of the passport with special characters like < and numbers.

STRICT REQUIREMENTS:
- Return EXACTLY 2 lines, each EXACTLY 44 characters long
- Preserve ALL < symbols and spacing exactly as they appear
- Do NOT add explanations, headers, formatting, or any extra text
- If MRZ is not clearly visible, return empty string
- If you're unsure, return empty string

VALID EXAMPLES:
P<POLMUSIELAK<<<BORYS<ANDRZEJ<<<<<<<<<<<<<<<
EM9638245<POL8404238M33012567544<<<<<<<02<<<

P<POLMUSIELAK<<BORYS<ANDRZEJ<<<<<<<<<<<<<<<<
EM9638245<POL8404238M33012567544<<<<<<<02<<<

P<POLMUSIELAK<<<BORYSANDRZEJ<<<<<<<<<<<<<<<<
EM9638245<POL8404238M33012567544<<<<<<<02<<<

INVALID RESPONSES (do not return these):
- Any text with explanations or formatting
- Responses that don't match the exact 2-line, 44-char format
- Partial or incomplete MRZ data

Return ONLY the MRZ text following the strict requirements above:"""
        
        def _make_api_call():
            model = _get_model()
            response = model.generate_content([prompt, image])
            if not response or not response.text:
                return ""
            return response.text.strip()
        
        # Make API call with retry logic
        response_text = _retry_api_call(_make_api_call, max_retries=max_retries)
        
        if not response_text:
            return ""
        
        # Clean and validate the response
        mrz_text = _clean_mrz_response(response_text)
        
        # Final validation - ensure we have exactly 2 lines of 44 characters each
        lines = mrz_text.split('\n')
        if len(lines) == 2 and all(len(line) == 44 for line in lines):
            return mrz_text
        elif len(lines) == 1 and len(lines[0]) == 44:
            # Single line found, try to find second line
            return mrz_text
        else:
            return ""
        
    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg or "api_key" in error_msg:
            raise GeminiOCRError(f"Invalid or missing Gemini API key. {error_msg}", "authentication")
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            raise GeminiOCRError(f"API quota exceeded. {error_msg}", "quota")
        elif "permission" in error_msg.lower():
            raise GeminiOCRError(f"API permission denied. {error_msg}", "permission")
        elif "rate limit" in error_msg.lower():
            raise GeminiOCRError(f"API rate limit exceeded. {error_msg}", "rate_limit")
        else:
            raise GeminiOCRError(f"API call failed: {error_msg}", "api_error")


def extract_text_with_gemini(image: Image.Image, max_retries: int = 2) -> Optional[str]:
    """
    Extract all text from passport image using Gemini Vision (Full text approach)
    
    This is the FALLBACK method when MRZ extraction fails
    
    Args:
        image: PIL Image object
        max_retries: Maximum number of API call retries (default: 2)
        
    Returns:
        Full text extracted from image or empty string if extraction fails
        
    Raises:
        GeminiOCRError: If API call fails with specific error information
    """
    try:
        _validate_api_key()
        
        # Enhanced prompt with structured requirements
        prompt = """Extract ALL visible text from this passport image with STRUCTURED FORMATTING.

STRICT REQUIREMENTS:
- Extract EVERY piece of visible text
- Preserve field labels exactly as they appear (Surname, Given Names, Passport Number, etc.)
- Preserve all data values with their exact formatting
- Include the complete MRZ (Machine Readable Zone) at the bottom
- Maintain the original structure, line breaks, and spacing
- Include all dates, country codes, document numbers, and personal information
- Do NOT add explanations, interpretations, or formatting comments
- Return the EXACT text as it appears in the image

STRUCTURED OUTPUT FORMAT:
Field Label: Field Value
Field Label: Field Value
...
MRZ Line 1
MRZ Line 2

EXAMPLE OUTPUT:
Surname: JARBOUAA
Given Names: ABDALLAH
Passport Number: 00000000
Country Code: SYR
Date of Birth: 01/01/1980
Date of Expiry: 01/01/2030
Sex: M
P<SYRJARBOUAA<<ABDALLAH<<<<<<<<<<<<<<<<<<<
00000000<SYR800101M3001010<<<<<<<<<<<<<<<0

Return the complete structured text extraction following the requirements above:"""
        
        def _make_api_call():
            model = _get_model()
            response = model.generate_content([prompt, image])
            if not response or not response.text:
                return ""
            return response.text.strip()
        
        # Make API call with retry logic
        response_text = _retry_api_call(_make_api_call, max_retries=max_retries)
        
        if not response_text:
            return ""
        
        # Clean response - remove any markdown formatting
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            clean_lines = [line for line in lines if not line.startswith('```') and line.strip()]
            response_text = '\n'.join(clean_lines).strip()
        
        return response_text
        
    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg or "api_key" in error_msg:
            raise GeminiOCRError(f"Invalid or missing Gemini API key. {error_msg}", "authentication")
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            raise GeminiOCRError(f"API quota exceeded. {error_msg}", "quota")
        elif "permission" in error_msg.lower():
            raise GeminiOCRError(f"API permission denied. {error_msg}", "permission")
        elif "rate limit" in error_msg.lower():
            raise GeminiOCRError(f"API rate limit exceeded. {error_msg}", "rate_limit")
        else:
            raise GeminiOCRError(f"API call failed: {error_msg}", "api_error")


def test_gemini_connection() -> bool:
    """
    Test if Gemini API is properly configured and accessible
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        _validate_api_key()
        
        # Try to list models to verify API key
        models = genai.list_models()
        print("[+] Gemini API connection successful")
        
        # Test model initialization
        try:
            model = _get_model()
            print(f"[+] Model '{model.model_name}' initialized successfully")
        except Exception as e:
            print(f"[!] Model initialization warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"[-] Gemini API connection failed: {e}")
        return False


def validate_mrz_format(mrz_text: str) -> Tuple[bool, str]:
    """
    Validate MRZ text format
    
    Args:
        mrz_text: MRZ text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not mrz_text:
        return False, "Empty MRZ text"
    
    lines = mrz_text.split('\n')
    if len(lines) != 2:
        return False, f"Expected 2 lines, got {len(lines)}"
    
    for i, line in enumerate(lines):
        if len(line) != 44:
            return False, f"Line {i+1} has {len(line)} characters, expected 44"
        
        # Check if line contains valid MRZ characters (letters, numbers, <)
        if not all(c.isalnum() or c == '<' for c in line):
            return False, f"Line {i+1} contains invalid characters"
    
    # Check if first line starts with P< (passport indicator)
    if not lines[0].startswith('P<'):
        return False, "First line should start with 'P<'"
    
    return True, "Valid MRZ format"


if __name__ == "__main__":
    # Test connection when module is run directly
    print("Testing Gemini API connection...")
    test_gemini_connection()
