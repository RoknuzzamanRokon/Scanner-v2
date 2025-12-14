"""
PDF utilities for converting PDF files to images
"""
import base64
import io
from typing import List
from PIL import Image
from pdf2image import convert_from_bytes
import tempfile
import os


def decode_base64_pdf(pdf_base64: str) -> bytes:
    """
    Decode base64 encoded PDF to bytes
    
    Args:
        pdf_base64: Base64 encoded PDF string
        
    Returns:
        PDF bytes
    """
    # Remove data URL prefix if present
    if ',' in pdf_base64:
        pdf_base64 = pdf_base64.split(',')[1]
    
    # Decode base64
    pdf_bytes = base64.b64decode(pdf_base64)
    return pdf_bytes


def convert_pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    """
    Convert PDF pages to PIL Images
    
    Args:
        pdf_bytes: PDF file as bytes
        dpi: Resolution for conversion (default 300 for OCR quality)
        
    Returns:
        List of PIL Images, one per page
        
    Raises:
        Exception: If poppler is not installed or conversion fails
    """
    try:
        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
        print(f"✓ Converted PDF to {len(images)} image(s)")
        return images
    except Exception as e:
        error_msg = str(e)
        
        # Check if it's a poppler-related error
        if 'poppler' in error_msg.lower() or 'Unable to get page count' in error_msg:
            raise Exception(
                "Poppler is not installed or not in PATH. "
                "\n\n"
                "INSTALLATION INSTRUCTIONS FOR WINDOWS:\n"
                "1. Download poppler from: https://github.com/oschwartz10612/poppler-windows/releases/latest\n"
                "2. Extract the ZIP file to C:\\poppler\n"
                "3. Add C:\\poppler\\Library\\bin to your system PATH\n"
                "4. Restart your terminal and try again\n"
                "\n"
                "Or run: INSTALL_POPPLER_WINDOWS.bat\n"
                "\n"
                f"Original error: {error_msg}"
            )
        else:
            print(f"✗ Error converting PDF to images: {e}")
            raise Exception(f"PDF conversion failed: {error_msg}")


def convert_pdf_base64_to_images(pdf_base64: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert base64 encoded PDF to PIL Images
    
    Args:
        pdf_base64: Base64 encoded PDF string
        dpi: Resolution for conversion (default 300 for OCR quality)
        
    Returns:
        List of PIL Images, one per page
    """
    try:
        # Decode base64 PDF
        pdf_bytes = decode_base64_pdf(pdf_base64)
        
        # Convert to images
        images = convert_pdf_to_images(pdf_bytes, dpi=dpi)
        return images
    except Exception as e:
        print(f"✗ Error converting base64 PDF to images: {e}")
        raise


def get_first_page_as_image(pdf_base64: str, dpi: int = 300) -> Image.Image:
    """
    Get the first page of a PDF as a PIL Image
    
    Args:
        pdf_base64: Base64 encoded PDF string
        dpi: Resolution for conversion (default 300 for OCR quality)
        
    Returns:
        PIL Image of the first page
    """
    images = convert_pdf_base64_to_images(pdf_base64, dpi=dpi)
    if not images:
        raise ValueError("PDF contains no pages")
    
    print(f"→ Using first page of PDF (total pages: {len(images)})")
    return images[0]


def is_pdf(data: str) -> bool:
    """
    Check if base64 encoded data is a PDF
    
    Args:
        data: Base64 encoded data
        
    Returns:
        True if data is a PDF, False otherwise
    """
    try:
        # Check for PDF header in base64
        if data.startswith('data:application/pdf'):
            return True
        
        # Decode and check magic number
        if ',' in data:
            data = data.split(',')[1]
        
        decoded = base64.b64decode(data[:100])  # Check first 100 bytes
        return decoded.startswith(b'%PDF')
    except:
        return False
