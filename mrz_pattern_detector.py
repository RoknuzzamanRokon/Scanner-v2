"""
MRZ Pattern Detector - Detects "P<" pattern to locate MRZ precisely
"""
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import pytesseract


def detect_mrz_by_pattern(image: Image.Image, search_region: Optional[Tuple] = None) -> Optional[Tuple[int, int]]:
    """
    Detect MRZ by finding "P<" pattern using OCR
    
    Args:
        image: PIL Image to search in
        search_region: (y_start, y_end) to limit search area, or None for full image
        
    Returns:
        (x, y) position of "P<" pattern, or None if not found
    """
    # Convert to OpenCV
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        gray = img_array
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    height, width = gray.shape
    
    # Apply search region if provided
    if search_region:
        y_start, y_end = search_region
        search_gray = gray[y_start:y_end, :]
        offset_y = y_start
    else:
        search_gray = gray
        offset_y = 0
    
    # Enhance image for better OCR
    # 1. Apply binary threshold
    _, binary = cv2.threshold(search_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Denoise
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    # 3. Upscale for better OCR
    scale = 2
    upscaled = cv2.resize(denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Use pytesseract to find text with positions
    # Use whitelist for MRZ characters
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
    
    try:
        # Get detailed OCR data (includes positions)
        data = pytesseract.image_to_data(upscaled, output_type=pytesseract.Output.DICT, config=custom_config)
        
        # Search for "P<" pattern
        for i, text in enumerate(data['text']):
            if text and text.strip().startswith('P<'):
                # Found "P<" pattern
                # Get position (scale back down from upscaled image)
                x = int(data['left'][i] / scale)
                y = int(data['top'][i] / scale)
                
                # Adjust for search region offset
                y_absolute = y + offset_y
                
                print(f"  → Found 'P<' pattern at position: x={x}, y={y_absolute}")
                return (x, y_absolute)
        
        # Try to find just "P" followed by "<" nearby
        for i in range(len(data['text']) - 1):
            text = data['text'][i]
            next_text = data['text'][i + 1]
            
            if text and text.strip() == 'P' and next_text and '<' in next_text:
                x = int(data['left'][i] / scale)
                y = int(data['top'][i] / scale)
                y_absolute = y + offset_y
                
                print(f"  → Found 'P' + '<' pattern at position: x={x}, y={y_absolute}")
                return (x, y_absolute)
    
    except Exception as e:
        print(f"  ⚠ OCR pattern detection failed: {e}")
    
    return None


def detect_mrz_zone_by_pattern(image: Image.Image, photo_region: Optional[Tuple] = None, save_debug: bool = False, image_url: str = "", user_folder = None) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect MRZ zone by finding "P<" pattern and selecting that line + next line
    
    Args:
        image: PIL Image (passport page)
        photo_region: (x, y, w, h) of photo, or None
        save_debug: Whether to save debug visualization
        image_url: URL of the image for AI fallback
        
    Returns:
        (x, y, w, h) of MRZ zone containing 2 lines, or None
    """

    # Convert to numpy
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        gray = img_array
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        color_img = img_array.copy()
    
    height, width = img_array.shape[0], img_array.shape[1]
    
    # Define search region (below photo if available)
    if photo_region:
        photo_x, photo_y, photo_w, photo_h = photo_region
        search_y_start = photo_y + photo_h
        search_y_end = height
    else:
        # Search bottom 40% of image
        search_y_start = int(height * 0.60)
        search_y_end = height
    
    print(f"  → Searching for 'P<' pattern in region y={search_y_start}-{search_y_end}...")
    
    # Get search region
    search_region = gray[search_y_start:search_y_end, :]
    
    # Enhance for OCR
    _, binary = cv2.threshold(search_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    # Find horizontal projection to detect text lines
    horizontal_projection = np.sum(denoised, axis=1)
    
    # Normalize
    if np.max(horizontal_projection) > 0:
        horizontal_projection = horizontal_projection / np.max(horizontal_projection)
    
    # Find rows with significant text
    text_threshold = 0.20  # Lowered to detect fainter lines
    text_rows = np.where(horizontal_projection > text_threshold)[0]
    
    if len(text_rows) == 0:
        print("  ✗ No text rows found")
        return None
    
    # Group consecutive rows into lines
    lines = []
    current_start = text_rows[0]
    current_end = text_rows[0]
    
    for i in range(1, len(text_rows)):
        if text_rows[i] - text_rows[i-1] <= 5:  # Allow slightly more gap
            current_end = text_rows[i]
        else:
            if current_end - current_start >= 10:  # Accept smaller lines
                lines.append((current_start, current_end))
            current_start = text_rows[i]
            current_end = text_rows[i]
    
    if current_end - current_start >= 10:
        lines.append((current_start, current_end))
    
    print(f"  → Found {len(lines)} text lines in search region")
    
    # Now use OCR to find which line contains "P<"
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
    
    mrz_line_index = None
    
    try:
        # Try OCR on each detected line
        for idx, (line_start, line_end) in enumerate(lines):
            line_img = search_region[line_start:line_end, :]
            
            # Upscale for better OCR
            line_upscaled = cv2.resize(line_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # OCR this line
            text = pytesseract.image_to_string(line_upscaled, config=custom_config).strip()
            
            # Check if this line starts with "P<"
            if text.startswith('P<') or text.startswith('P <'):
                print(f"  ✓ Found 'P<' in line {idx + 1}: {text[:30]}...")
                mrz_line_index = idx
                break
    
    except Exception as e:
        print(f"  ⚠ OCR failed: {e}")
    
    # If "P<" found, select that line + next line (2 MRZ lines)
    if mrz_line_index is not None and mrz_line_index < len(lines) - 1:
        # Get the line with "P<" and the next line
        line1_start, line1_end = lines[mrz_line_index]
        line2_start, line2_end = lines[mrz_line_index + 1]
        
        # MRZ zone: from start of first line to end of second line
        mrz_y = search_y_start + line1_start
        mrz_bottom = search_y_start + line2_end
        mrz_h = mrz_bottom - mrz_y
        
        # Add small padding
        padding = 5
        mrz_y = max(0, mrz_y - padding)
        mrz_h = min(height - mrz_y, mrz_h + 2 * padding)
        
        print(f"  ✓ MRZ zone: 2 lines detected (y={mrz_y}-{mrz_y + mrz_h}, height={mrz_h}px)")
        
        # Create debug visualization
        if save_debug:
            debug_img = color_img.copy()
            
            # Draw both MRZ lines
            line1_y_abs = search_y_start + line1_start
            line1_h = line1_end - line1_start
            line2_y_abs = search_y_start + line2_start
            line2_h = line2_end - line2_start
            
            # Draw line 1 (P< line) in red
            cv2.rectangle(debug_img, (0, line1_y_abs), (width, line1_y_abs + line1_h), (0, 0, 255), 2)
            cv2.putText(debug_img, "MRZ LINE 1 (P<)", (10, line1_y_abs - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw line 2 in blue
            cv2.rectangle(debug_img, (0, line2_y_abs), (width, line2_y_abs + line2_h), (255, 0, 0), 2)
            cv2.putText(debug_img, "MRZ LINE 2", (10, line2_y_abs - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw full MRZ zone in green
            cv2.rectangle(debug_img, (0, mrz_y), (width, mrz_y + mrz_h), (0, 255, 0), 3)
            cv2.putText(debug_img, f"MRZ ZONE ({mrz_h}px)", (10, mrz_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save debug image
            debug_pil = Image.fromarray(debug_img)
            from pathlib import Path
            if user_folder:
                debug_save_path = Path(user_folder) / 'mrz_pattern_debug.jpg'
            else:
                debug_save_path = Path('temp') / 'mrz_pattern_debug.jpg'
            debug_pil.save(str(debug_save_path))
            print(f"  ✓ Saved debug: {debug_save_path}")
        
        return (0, mrz_y, width, mrz_h)
    
    else:
        print("  ✗ Could not find 'P<' pattern or missing next line")
        


def get_mrz_line_coordinates(image: Image.Image, photo_region: Optional[Tuple] = None):
    """
    Get individual MRZ line coordinates for drawing on debug image
    
    Returns:
        {
            'line1': (x, y, w, h) or None,
            'line2': (x, y, w, h) or None,
            'mrz_zone': (x, y, w, h) or None
        }
    """
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        gray = img_array
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    height, width = img_array.shape[0], img_array.shape[1]
    
    # Define search region
    if photo_region:
        photo_x, photo_y, photo_w, photo_h = photo_region
        search_y_start = photo_y + photo_h
        search_y_end = height
    else:
        search_y_start = int(height * 0.60)
        search_y_end = height
    
    search_region = gray[search_y_start:search_y_end, :]
    
    # Enhance for OCR
    _, binary = cv2.threshold(search_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    # Find horizontal projection
    horizontal_projection = np.sum(denoised, axis=1)
    if np.max(horizontal_projection) > 0:
        horizontal_projection = horizontal_projection / np.max(horizontal_projection)
    
    # Find text rows
    text_threshold = 0.20  # Lowered from 0.30 to detect fainter lines
    text_rows = np.where(horizontal_projection > text_threshold)[0]
    
    if len(text_rows) == 0:
        return {'line1': None, 'line2': None, 'mrz_zone': None}
    
    # Group into lines
    lines = []
    current_start = text_rows[0]
    current_end = text_rows[0]
    
    for i in range(1, len(text_rows)):
        if text_rows[i] - text_rows[i-1] <= 5:  # Increased from 3 to allow slightly more gaps
            current_end = text_rows[i]
        else:
            if current_end - current_start >= 10:  # Reduced from 15 to accept smaller lines
                lines.append((current_start, current_end))
            current_start = text_rows[i]
            current_end = text_rows[i]
    
    if current_end - current_start >= 10:  # Reduced from 15
        lines.append((current_start, current_end))
    
    # Try OCR to find "P<" line
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
    mrz_line_index = None
    
    try:
        for idx, (line_start, line_end) in enumerate(lines):
            line_img = search_region[line_start:line_end, :]
            line_upscaled = cv2.resize(line_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(line_upscaled, config=custom_config).strip()
            
            if text.startswith('P<') or text.startswith('P <'):
                mrz_line_index = idx
                break
    except Exception as e:
        pass
    
    # Return line coordinates
    if mrz_line_index is not None and mrz_line_index < len(lines) - 1:
        line1_start, line1_end = lines[mrz_line_index]
        line2_start, line2_end = lines[mrz_line_index + 1]
        
        # Convert to absolute coordinates
        line1_y = search_y_start + line1_start
        line1_h = line1_end - line1_start
        line2_y = search_y_start + line2_start
        line2_h = line2_end - line2_start
        
        # MRZ zone
        mrz_y = line1_y - 5
        mrz_h = (line2_y + line2_h) - line1_y + 10
        
        return {
            'line1': (0, line1_y, width, line1_h),
            'line2': (0, line2_y, width, line2_h),
            'mrz_zone': (0, mrz_y, width, mrz_h)
        }
    
    return {'line1': None, 'line2': None, 'mrz_zone': None}

