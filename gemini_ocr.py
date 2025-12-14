"""
Gemini AI OCR integration for passport data extraction
"""
import os
import google.generativeai as genai
from PIL import Image
from typing import Optional
from config import config


# Configure Gemini API
if config.GEMINI_API_KEY:
    genai.configure(api_key=config.GEMINI_API_KEY)
else:
    print("⚠ Warning: GEMINI_API_KEY not found in environment variables")


def extract_mrz_with_gemini(image: Image.Image) -> Optional[str]:
    """
    Extract MRZ text from passport image using Gemini Vision (Optimized approach)
    
    This is the PRIORITY method for AI extraction - faster and more focused
    
    Args:
        image: PIL Image object
        
    Returns:
        MRZ text (2 lines) or empty string if extraction fails
        
    Raises:
        Exception: If API call fails
    """
    try:
        if not config.GEMINI_API_KEY:
            raise Exception("AI parser failed: GEMINI_API_KEY not configured")
        
        # Use Gemini Flash for faster processing
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = """Extract ONLY the Machine Readable Zone (MRZ) text from this passport image.

The MRZ is the 2-line text at the bottom of the passport with special characters like < and numbers.

Requirements:
- Return EXACTLY 2 lines
- Each line must be EXACTLY 44 characters
- Preserve all < symbols and spacing
- Do not add any explanations, headers, or formatting
- If you cannot find the MRZ, return empty

Example format:
P<POLMUSIELAK<<<BORYS<ANDRZEJ<<<<<<<<<<<<<<<
EM9638245<POL8404238M33012567544<<<<<<<02<<<

Return only the MRZ text, nothing else:"""
        
        # Generate content
        response = model.generate_content([prompt, image])
        
        if not response or not response.text:
            return ""
        
        # Clean the response
        mrz_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if mrz_text.startswith('```'):
            lines = mrz_text.split('\n')
            mrz_text = '\n'.join([line for line in lines if not line.startswith('```')])
            mrz_text = mrz_text.strip()
        
        # Validate format
        lines = mrz_text.split('\n')
        if len(lines) >= 2:
            # Return only first 2 lines
            return f"{lines[0]}\n{lines[1]}"
        
        return ""
        
    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg or "api_key" in error_msg:
            raise Exception(f"AI parser failed: Invalid or missing Gemini API key. {error_msg}")
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            raise Exception(f"AI parser failed: API quota exceeded. {error_msg}")
        elif "permission" in error_msg.lower():
            raise Exception(f"AI parser failed: API permission denied. {error_msg}")
        else:
            raise Exception(f"AI parser failed: {error_msg}")


def extract_text_with_gemini(image: Image.Image) -> Optional[str]:
    """
    Extract all text from passport image using Gemini Vision (Full text approach)
    
    This is the FALLBACK method when MRZ extraction fails
    
    Args:
        image: PIL Image object
        
    Returns:
        Full text extracted from image or empty string if extraction fails
        
    Raises:
        Exception: If API call fails
    """
    try:
        if not config.GEMINI_API_KEY:
            raise Exception("AI parser failed: GEMINI_API_KEY not configured")
        
        # Use Gemini Flash for faster processing
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = """Extract ALL visible text from this passport image.

Requirements:
- Extract every piece of text you can see
- Preserve field labels (Surname, Given Names, Passport Number, etc.)
- Preserve all data values
- Include the MRZ (Machine Readable Zone) at the bottom
- Maintain the structure and line breaks
- Include dates, country codes, and all other information

Return the complete text extraction:"""
        
        # Generate content
        response = model.generate_content([prompt, image])
        
        if not response or not response.text:
            return ""
        
        return response.text.strip()
        
    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg or "api_key" in error_msg:
            raise Exception(f"AI parser failed: Invalid or missing Gemini API key. {error_msg}")
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            raise Exception(f"AI parser failed: API quota exceeded. {error_msg}")
        elif "permission" in error_msg.lower():
            raise Exception(f"AI parser failed: API permission denied. {error_msg}")
        else:
            raise Exception(f"AI parser failed: {error_msg}")


def test_gemini_connection() -> bool:
    """
    Test if Gemini API is properly configured and accessible
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        if not config.GEMINI_API_KEY:
            print("✗ GEMINI_API_KEY not configured")
            return False
        
        # Try to list models to verify API key
        models = genai.list_models()
        print("✓ Gemini API connection successful")
        return True
        
    except Exception as e:
        print(f"✗ Gemini API connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test connection when module is run directly
    print("Testing Gemini API connection...")
    test_gemini_connection()
