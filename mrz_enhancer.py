"""
Enhanced MRZ detection and cropping utilities
"""
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple


def enhance_mrz_for_tesseract(image: Image.Image) -> Image.Image:
    """
    Enhance MRZ region for Tesseract OCR (grayscale only, no binary)
    
    Args:
        image: PIL Image of MRZ region
        
    Returns:
        Enhanced PIL Image in grayscale
    """
    # Convert to OpenCV
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Upscale for better OCR
    height, width = gray.shape
    scale_factor = 2.5
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Light denoising
    denoised = cv2.fastNlMeansDenoising(resized, h=8)
    
    # Moderate CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Stay in grayscale - NO binary conversion
    return Image.fromarray(enhanced)
