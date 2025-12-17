"""
Tesseract OCR fallback validation with MRZ reconstruction and TD3 compliance
"""
import cv2
import numpy as np
import re
import tempfile
import os
from PIL import Image
from typing import Dict
import pytesseract
from country_code import get_country_info
from utils import format_mrz_date


def validate_passport_with_tesseract_fallback(image: Image.Image, verbose: bool = True) -> Dict:
    """
    Tesseract OCR Fallback Validation with Enhanced Preprocessing
    
    - Advanced image preprocessing for passport documents
    - MRZ zone detection and isolation
    - Multiple OCR attempts with different configurations
    - MRZ pattern detection and reconstruction
    - TD3 format validation & cleaning
    
    Args:
        image: PIL Image object
        verbose: Print detailed logs
        
    Returns:
        Dictionary with success status and passport data
    """
    try:
        if verbose:
            print(f"  → Processing with Tesseract OCR...")
        
        # Set Tesseract path from environment
        from config import config
        if hasattr(config, 'TESSERACT_CMD') and config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
            if verbose:
                print(f"  → Using Tesseract: {config.TESSERACT_CMD}")
        
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if verbose:
            print(f"  → Applying advanced image preprocessing...")
        
        # Try multiple preprocessing approaches
        processed_images = []
        
        # Approach 1: Enhanced preprocessing for passport documents
        processed_images.append(preprocess_for_passport_ocr(img_cv, method="enhanced"))
        
        # Approach 2: MRZ-specific preprocessing
        processed_images.append(preprocess_for_passport_ocr(img_cv, method="mrz_focused"))
        
        # Approach 3: High contrast preprocessing
        processed_images.append(preprocess_for_passport_ocr(img_cv, method="high_contrast"))
        
        # Approach 4: Detect and extract MRZ region specifically
        mrz_region = detect_mrz_region(img_cv)
        if mrz_region is not None:
            processed_images.append(preprocess_for_passport_ocr(mrz_region, method="mrz_only"))
        
        if verbose:
            print(f"  → Generated {len(processed_images)} preprocessed variants")
        
        best_result = None
        best_confidence = 0
        
        # Try OCR with different configurations on each preprocessed image
        ocr_configs = [
            # MRZ-specific config with character whitelist
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<',
            # Single text line mode for MRZ
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<',
            # Single word mode
            r'--oem 3 --psm 8',
            # Raw line mode
            r'--oem 3 --psm 13',
            # Default with character restriction
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
        ]
        
        for i, processed_img in enumerate(processed_images):
            for j, config in enumerate(ocr_configs):
                try:
                    if verbose:
                        print(f"    → Trying OCR variant {i+1}.{j+1}...")
                    
                    extracted_text = pytesseract.image_to_string(processed_img, config=config)
                    
                    if not extracted_text or len(extracted_text.strip()) < 10:
                        continue
                    
                    # Evaluate the quality of extracted text
                    text_quality = evaluate_ocr_quality(extracted_text, verbose=False)
                    
                    if text_quality > best_confidence:
                        best_confidence = text_quality
                        best_result = extracted_text
                        if verbose:
                            print(f"      ✓ New best result (quality: {text_quality:.2f})")
                    
                    # If we get a very good result, we can stop early
                    if text_quality > 0.8:
                        if verbose:
                            print(f"      ✓ High quality result found, stopping search")
                        break
                        
                except Exception as e:
                    if verbose:
                        print(f"      ✗ OCR variant {i+1}.{j+1} failed: {e}")
                    continue
            
            # Break outer loop if high quality found
            if best_confidence > 0.8:
                break
        
        if not best_result:
            if verbose:
                print(f"  ✗ Tesseract: No meaningful text detected in any variant")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "method_used": "Tesseract",
                "error": "No meaningful text detected by Tesseract"
            }
        
        extracted_text = best_result
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            if verbose:
                print(f"  ✗ Tesseract: No meaningful text detected")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "method_used": "Tesseract",
                "error": "No meaningful text detected by Tesseract"
            }
        
        # Clean and process extracted text
        lines = extracted_text.strip().split('\n')
        all_text = ' '.join([line.strip() for line in lines if line.strip()])
        
        if verbose:
            print(f"  ✓ Text extracted: {len(lines)} lines")
            print(f"    Text preview: {all_text}...")
        
        # Check if this looks like a passport with more flexible criteria
        passport_indicators = [
            "PASSPORT" in all_text.upper(),
            "REPUBLIC" in all_text.upper(),
            "PEOPLE" in all_text.upper(),
            "BANGLADESH" in all_text.upper(),
            "INDIA" in all_text.upper(),
            "PAKISTAN" in all_text.upper(),
            "USA" in all_text.upper(),
            bool(re.search(r"<{2,}", all_text)),  # MRZ padding (2+ < symbols)
            bool(re.search(r"P<[A-Z]{3}", all_text)),  # MRZ passport type
            bool(re.search(r"[A-Z0-9]{8,}", all_text)),  # Long alphanumeric sequences
            bool(re.search(r"\d{6}", all_text)),  # Date patterns
            len([line for line in lines if len(line.strip()) > 30]) >= 2,  # Long lines (potential MRZ)
        ]
        
        passport_score = sum(passport_indicators)
        
        if verbose:
            print(f"  → Passport indicators found: {passport_score}/12")
            if passport_score > 0:
                print(f"    Indicators: {[i for i, x in enumerate(passport_indicators, 1) if x]}")
        
        # Lower threshold - if we have any indicators, try to process
        if passport_score == 0:
            if verbose:
                print(f"  ✗ No passport indicators found in text")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "method_used": "Tesseract",
                "error": "No passport indicators found in extracted text"
            }
        
        # Try to extract MRZ lines with more flexible criteria
        mrz_lines = []
        potential_mrz = []
        
        # Look for lines that could be MRZ with multiple criteria
        for line in lines:
            clean_line = line.strip().upper()
            
            # MRZ criteria (any of these makes it a potential MRZ line):
            mrz_criteria = [
                '<' in clean_line and len(clean_line) >= 20,  # Traditional MRZ with < symbols
                len(clean_line) >= 35 and bool(re.search(r'[A-Z0-9]{10,}', clean_line)),  # Long alphanumeric
                clean_line.startswith('P') and len(clean_line) >= 30,  # Starts with P (passport type)
                bool(re.search(r'\d{6,}', clean_line)) and len(clean_line) >= 25,  # Contains dates
                len(clean_line) >= 40,  # Very long lines (likely MRZ)
            ]
            
            if any(mrz_criteria):
                potential_mrz.append(clean_line)
        
        if verbose:
            print(f"  → Found {len(potential_mrz)} potential MRZ segments")
            for i, mrz in enumerate(potential_mrz):
                print(f"    MRZ {i+1}: {mrz}")
        
        # Try to reconstruct MRZ from potential segments
        if len(potential_mrz) >= 2:
            # Find the correct order for TD3 format
            # Line 1 should start with 'P' (passport type)
            # Line 2 should contain passport number and other data
            
            line1_candidate = None
            line2_candidate = None
            
            # Look for line starting with 'P'
            for mrz in potential_mrz:
                if mrz.startswith('P'):
                    line1_candidate = mrz
                    break
            
            # Find the second line - should be the one that looks like a passport number line
            # TD3 Line 2 has specific characteristics: passport number + check digits + dates
            remaining_lines = [mrz for mrz in potential_mrz if mrz != line1_candidate]
            if remaining_lines:
                # Look for line that has MRZ characteristics:
                # 1. Contains multiple < symbols (MRZ padding)
                # 2. Has numeric patterns (dates, check digits)
                # 3. Is around 44 characters or can be padded to 44
                best_score = -1
                for line in remaining_lines:
                    score = 0
                    
                    # Score based on MRZ characteristics
                    if '<' in line:
                        score += line.count('<') * 2  # More < symbols = more likely MRZ
                    
                    # Check for numeric patterns (dates, passport numbers)
                    if re.search(r'\d{6,}', line):  # 6+ consecutive digits (dates)
                        score += 10
                    
                    # Check for typical MRZ length or close to it
                    if 40 <= len(line) <= 50:
                        score += 5
                    
                    # Penalize lines with common words (not MRZ)
                    if any(word in line.upper() for word in ['REPUBLIC', 'PEOPLE', 'BANGLADESH', 'PASSPORT']):
                        score -= 20
                    
                    if verbose:
                        print(f"      Line '{line[:30]}...' score: {score}")
                    
                    if score > best_score:
                        best_score = score
                        line2_candidate = line
                
                # If no good candidate found, use the longest remaining
                if not line2_candidate:
                    line2_candidate = max(remaining_lines, key=len)
            
            # Fallback: if no 'P' line found, use the two longest
            if not line1_candidate or not line2_candidate:
                potential_mrz.sort(key=len, reverse=True)
                line1_candidate = potential_mrz[0]
                line2_candidate = potential_mrz[1]
            
            # Clean and validate
            line1 = clean_mrz_line(line1_candidate)
            line2 = clean_mrz_line(line2_candidate)
            
            if len(line1) == 44 and len(line2) == 44:
                mrz_lines = [line1, line2]
            elif len(line1) >= 40 and len(line2) >= 40:
                # Pad to 44 characters
                line1 = line1[:44].ljust(44, '<')
                line2 = line2[:44].ljust(44, '<')
                mrz_lines = [line1, line2]
        
        # If we couldn't get proper MRZ lines, try alternative extraction methods
        if not mrz_lines:
            if verbose:
                print(f"  → No valid MRZ lines found, trying alternative extraction...")
            
            # Method 1: Try to extract MRZ patterns from raw text
            raw_mrz_candidates = extract_mrz_from_raw_text(all_text, verbose)
            
            if len(raw_mrz_candidates) >= 2:
                if verbose:
                    print(f"  → Found {len(raw_mrz_candidates)} MRZ candidates from raw text")
                
                # Try to form valid MRZ from candidates
                line1_candidate = None
                line2_candidate = None
                
                # Look for passport type line (starts with P)
                for candidate in raw_mrz_candidates:
                    if candidate.startswith('P') and len(candidate) >= 30:
                        line1_candidate = candidate
                        break
                
                # Find best second line
                remaining = [c for c in raw_mrz_candidates if c != line1_candidate]
                if remaining:
                    # Score remaining candidates for line 2 characteristics
                    best_score = -1
                    for candidate in remaining:
                        score = 0
                        if re.search(r'\d{6,}', candidate):  # Contains dates
                            score += 10
                        if len(candidate) >= 35:  # Good length
                            score += 5
                        if '<' in candidate:  # Has MRZ padding
                            score += candidate.count('<')
                        
                        if score > best_score:
                            best_score = score
                            line2_candidate = candidate
                
                if line1_candidate and line2_candidate:
                    # Clean and format the lines
                    line1 = clean_mrz_line(line1_candidate)
                    line2 = clean_mrz_line(line2_candidate)
                    
                    # Pad to 44 characters if needed
                    if len(line1) >= 35:
                        line1 = line1[:44].ljust(44, '<')
                    if len(line2) >= 35:
                        line2 = line2[:44].ljust(44, '<')
                    
                    if len(line1) == 44 and len(line2) == 44:
                        mrz_lines = [line1, line2]
                        if verbose:
                            print(f"  ✓ Formed MRZ from raw text candidates")
            
            # Method 2: Extract passport data from full text and reconstruct MRZ
            if not mrz_lines:
                if verbose:
                    print(f"  → Attempting data extraction from full text...")
                
                passport_data = extract_passport_data_from_text(all_text, verbose)
                
                if passport_data.get("passport_number") or passport_data.get("surname"):
                    # Try to reconstruct MRZ from extracted data
                    reconstructed_mrz = reconstruct_mrz_from_data(passport_data, verbose)
                    
                    if reconstructed_mrz:
                        mrz_lines = reconstructed_mrz.split('\n')
                        if verbose:
                            print(f"  ✓ MRZ reconstructed from extracted data")
            
            # If still no MRZ, return failure
            if not mrz_lines:
                if verbose:
                    print(f"  ✗ Could not extract or reconstruct valid MRZ")
                return {
                    "success": False,
                    "passport_data": passport_data if 'passport_data' in locals() else {},
                    "mrz_text": "",
                    "method_used": "Tesseract",
                    "error": "Could not extract or reconstruct valid MRZ from text"
                }
        
        # Validate MRZ using TD3 validation checker
        mrz_text = '\n'.join(mrz_lines)
        
        
        if verbose:
            print(f"  → Validating MRZ with TD3 rules...")
            print(f"    Line 1: {mrz_lines[0]}")
            print(f"    Line 2: {mrz_lines[1]}")
        
        from passport_detector import passport_validation_checker
        
        validation_result = passport_validation_checker(mrz_text, verbose=False)
        is_valid = validation_result.get("is_valid", False)
        confidence = validation_result.get("confidence_score", 0.0)
        
        if verbose:
            print(f"    TD3 Valid: {is_valid}")
            print(f"    Confidence: {confidence*100:.1f}%")
        
        # Reject if validation fails or confidence is too low
        if not is_valid or confidence < 0.5:
            if verbose:
                print(f"  ✗ MRZ validation failed: {validation_result.get('reason', 'Low confidence')}")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": mrz_text,
                "method_used": "Tesseract",
                "error": f"MRZ validation failed: {validation_result.get('reason', 'Low confidence')} (confidence: {confidence*100:.1f}%)"
            }
        
        # Extract passport data from validated MRZ
        passport_data = validation_result.get("passport_data", {})
        
        # Enhance with country information
        country_code = passport_data.get("country_code", "")
        if country_code:
            country_info = get_country_info(country_code)
            passport_data.update({
                "country_name": country_info.get("name", country_code),
                "country_iso": country_info.get("alpha2", ""),
                "nationality": country_info.get("nationality", country_code)
            })
        
        # Format dates
        if passport_data.get("date_of_birth"):
            passport_data["date_of_birth"] = format_mrz_date(passport_data["date_of_birth"])
        if passport_data.get("expiry_date"):
            passport_data["expiry_date"] = format_mrz_date(passport_data["expiry_date"])
        
        if verbose:
            print(f"  ✓ Passport data extracted successfully")
            print(f"    Surname: {passport_data.get('surname', '')}")
            print(f"    Given Names: {passport_data.get('given_names', '')}")
            print(f"    Passport #: {passport_data.get('passport_number', '')}")
            print(f"    Country: {passport_data.get('country_name', '')} ({country_code})")
        
        return {
            "success": True,
            "passport_data": passport_data,
            "mrz_text": mrz_text,
            "method_used": "Tesseract",
            "error": ""
        }
    
    except Exception as e:
        import traceback
        if verbose:
            print(f"  ✗ Tesseract error: {e}")
            traceback.print_exc()
        
        return {
            "success": False,
            "passport_data": {},
            "mrz_text": "",
            "method_used": "Tesseract",
            "error": f"Tesseract processing error: {str(e)}"
        }


def preprocess_for_passport_ocr(img: np.ndarray, method: str = "enhanced") -> np.ndarray:
    """
    Advanced preprocessing for passport OCR with multiple methods
    
    Args:
        img: OpenCV image (BGR format)
        method: Preprocessing method ("enhanced", "mrz_focused", "high_contrast", "mrz_only")
        
    Returns:
        Preprocessed image ready for OCR
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    if method == "enhanced":
        # Enhanced preprocessing for general passport text
        
        # 1. Upscale for better character recognition
        height, width = gray.shape
        gray = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        
        # 2. Noise reduction
        gray = cv2.medianBlur(gray, 3)
        
        # 3. Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 4. Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # 5. Adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        
        return binary
    
    elif method == "mrz_focused":
        # Preprocessing specifically optimized for MRZ text
        
        # 1. Aggressive upscaling for small MRZ text
        height, width = gray.shape
        gray = cv2.resize(gray, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
        
        # 2. Gaussian blur to smooth out noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 3. Sharpen the image to enhance text edges
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel_sharpen)
        
        # 4. Binary threshold with Otsu's method
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    elif method == "high_contrast":
        # High contrast preprocessing for difficult images
        
        # 1. Moderate upscaling
        height, width = gray.shape
        gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
        
        # 2. Histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # 3. Apply bilateral filter to reduce noise while keeping edges sharp
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 4. Simple binary threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 5. Dilate slightly to make text thicker
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        return binary
    
    elif method == "mrz_only":
        # Preprocessing for isolated MRZ region
        
        # 1. Significant upscaling since MRZ is usually small
        height, width = gray.shape
        gray = cv2.resize(gray, (width * 5, height * 5), interpolation=cv2.INTER_CUBIC)
        
        # 2. Enhance contrast specifically for MRZ
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        
        # 3. Gaussian blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 4. Adaptive threshold with larger block size for MRZ
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
        
        # 5. Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    else:
        # Default: simple preprocessing
        gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2))
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary


def detect_mrz_region(img: np.ndarray) -> np.ndarray:
    """
    Detect and extract the MRZ region from passport image
    
    Args:
        img: OpenCV image (BGR format)
        
    Returns:
        Cropped MRZ region or None if not found
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection to find text regions
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular regions that could be MRZ
        # MRZ is typically at the bottom of passport and has specific aspect ratio
        height, width = gray.shape
        
        potential_mrz_regions = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # MRZ characteristics:
            # - Usually in bottom 40% of image
            # - Width should be significant portion of image width
            # - Height should be relatively small (2 lines of text)
            # - Aspect ratio should be wide (width >> height)
            
            if (y > height * 0.6 and  # In bottom 40%
                w > width * 0.4 and   # At least 40% of image width
                h > 20 and h < height * 0.15 and  # Reasonable height for 2 text lines
                w / h > 5):  # Wide aspect ratio
                
                potential_mrz_regions.append((x, y, w, h, w * h))  # Include area for sorting
        
        if not potential_mrz_regions:
            return None
        
        # Sort by area (largest first) and take the best candidate
        potential_mrz_regions.sort(key=lambda x: x[4], reverse=True)
        
        # Extract the most likely MRZ region
        x, y, w, h, _ = potential_mrz_regions[0]
        
        # Add some padding around the detected region
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        mrz_region = gray[y:y+h, x:x+w]
        
        return mrz_region
    
    except Exception:
        return None


def extract_mrz_from_raw_text(text: str, verbose: bool = False) -> list:
    """
    Try to extract MRZ-like patterns from raw OCR text
    Even when < symbols are not properly detected
    
    Args:
        text: Raw OCR text
        verbose: Print debug info
        
    Returns:
        List of potential MRZ lines
    """
    lines = text.strip().split('\n')
    potential_mrz = []
    
    if verbose:
        print(f"    → Analyzing {len(lines)} lines for MRZ patterns...")
    
    for i, line in enumerate(lines):
        clean_line = line.strip().upper()
        
        if len(clean_line) < 20:
            continue
        
        # Score each line based on MRZ characteristics
        score = 0
        reasons = []
        
        # Length scoring (MRZ lines are typically 44 characters)
        if 40 <= len(clean_line) <= 50:
            score += 10
            reasons.append("good_length")
        elif 30 <= len(clean_line) <= 60:
            score += 5
            reasons.append("ok_length")
        
        # Character pattern scoring
        if re.search(r'P[A-Z]{3}', clean_line):  # P + country code
            score += 15
            reasons.append("passport_type")
        
        if re.search(r'[A-Z0-9]{8,}', clean_line):  # Long alphanumeric sequences
            score += 8
            reasons.append("long_alphanum")
        
        if re.search(r'\d{6}', clean_line):  # Date patterns
            score += 10
            reasons.append("date_pattern")
        
        if '<' in clean_line:
            score += clean_line.count('<') * 2  # More < = more likely MRZ
            reasons.append("mrz_padding")
        
        # Position scoring (MRZ usually at bottom)
        if i > len(lines) * 0.7:  # In bottom 30% of text
            score += 5
            reasons.append("bottom_position")
        
        # Penalize lines with common words (not MRZ)
        common_words = ['REPUBLIC', 'PEOPLE', 'PASSPORT', 'GOVERNMENT', 'MINISTRY']
        word_penalty = sum(2 for word in common_words if word in clean_line)
        score -= word_penalty
        if word_penalty > 0:
            reasons.append(f"word_penalty_{word_penalty}")
        
        if verbose and score > 5:
            print(f"      Line {i+1}: '{clean_line[:50]}...' score={score} ({', '.join(reasons)})")
        
        if score >= 10:  # Threshold for potential MRZ
            potential_mrz.append((score, clean_line, i))
    
    # Sort by score and return top candidates
    potential_mrz.sort(key=lambda x: x[0], reverse=True)
    
    if verbose:
        print(f"    → Found {len(potential_mrz)} potential MRZ lines")
    
    return [line for score, line, idx in potential_mrz[:5]]  # Return top 5


def evaluate_ocr_quality(text: str, verbose: bool = False) -> float:
    """
    Evaluate the quality of OCR extracted text for passport documents
    
    Args:
        text: Extracted text from OCR
        verbose: Print evaluation details
        
    Returns:
        Quality score between 0.0 and 1.0
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    
    score = 0.0
    factors = []
    
    # Factor 1: Presence of passport-related keywords
    passport_keywords = ['PASSPORT', 'REPUBLIC', 'PEOPLE', 'BANGLADESH', 'INDIA', 'USA', 'PAKISTAN']
    keyword_count = sum(1 for keyword in passport_keywords if keyword in text.upper())
    keyword_score = min(keyword_count * 0.2, 0.4)
    score += keyword_score
    factors.append(f"Keywords: {keyword_score:.2f}")
    
    # Factor 2: Presence of MRZ-like patterns
    mrz_patterns = [
        r'P<[A-Z]{3}',  # Passport type and country
        r'<{2,}',       # MRZ padding
        r'[A-Z0-9]{6,}', # Long alphanumeric sequences
        r'\d{6}',       # Date patterns
    ]
    
    mrz_score = 0
    for pattern in mrz_patterns:
        if re.search(pattern, text):
            mrz_score += 0.1
    
    mrz_score = min(mrz_score, 0.3)
    score += mrz_score
    factors.append(f"MRZ patterns: {mrz_score:.2f}")
    
    # Factor 3: Text readability (ratio of readable characters)
    total_chars = len(text)
    readable_chars = len(re.findall(r'[A-Za-z0-9<>\s]', text))
    readability = readable_chars / total_chars if total_chars > 0 else 0
    readability_score = readability * 0.2
    score += readability_score
    factors.append(f"Readability: {readability_score:.2f}")
    
    # Factor 4: Penalize excessive garbage characters
    garbage_chars = len(re.findall(r'[^\w\s<>]', text))
    garbage_penalty = min(garbage_chars / total_chars * 0.3, 0.3) if total_chars > 0 else 0
    score -= garbage_penalty
    factors.append(f"Garbage penalty: -{garbage_penalty:.2f}")
    
    # Factor 5: Bonus for structured text (multiple lines, proper spacing)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) >= 2:
        structure_bonus = 0.1
        score += structure_bonus
        factors.append(f"Structure bonus: {structure_bonus:.2f}")
    
    # Normalize score to 0-1 range
    final_score = max(0.0, min(1.0, score))
    
    if verbose:
        print(f"    OCR Quality Evaluation: {final_score:.2f}")
        for factor in factors:
            print(f"      {factor}")
    
    return final_score


def clean_mrz_line(line: str) -> str:
    """
    Clean and normalize MRZ line for Tesseract output
    
    Args:
        line: Raw MRZ line text from Tesseract
        
    Returns:
        Cleaned MRZ line
    """
    # Remove spaces and convert to uppercase
    cleaned = line.replace(' ', '').upper()
    
    # Replace common Tesseract OCR errors
    replacements = {
        '0': 'O',  # Zero to O in names
        '1': 'I',  # One to I in names  
        '8': 'B',  # Eight to B in names (sometimes)
        '5': 'S',  # Five to S in names (sometimes)
        '6': 'G',  # Six to G in names (sometimes)
        '2': 'Z',  # Two to Z in names (sometimes)
    }
    
    # Only apply replacements in name sections (not in dates/numbers)
    if cleaned.startswith('P<'):
        # This is line 1 (names), apply letter replacements carefully
        for old, new in replacements.items():
            if old in cleaned[5:]:  # Only in name section
                cleaned = cleaned[:5] + cleaned[5:].replace(old, new)
    
    return cleaned


def extract_passport_data_from_text(text: str, verbose: bool = False) -> Dict:
    """
    Extract passport data from full text using pattern matching
    (Same as EasyOCR version but optimized for Tesseract output)
    """
    data = {}
    
    # Passport number patterns (more flexible for Tesseract)
    passport_patterns = [
        r'(?:PASSPORT\s+(?:NO|NUMBER|#)\.?\s*:?\s*)([A-Z0-9]+)',
        r'(?:DOCUMENT\s+(?:NO|NUMBER|#)\.?\s*:?\s*)([A-Z0-9]+)',
        r'(?:NO\.?\s*:?\s*)([A-Z0-9]{6,12})',
        r'([A-Z]{2}\d{7})',  # Common passport number format
    ]
    
    for pattern in passport_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["passport_number"] = match.group(1)
            break
    
    # Name patterns
    name_patterns = [
        r'(?:SURNAME|FAMILY\s+NAME)\s*:?\s*([A-Z\s]+?)(?:\s+GIVEN|$)',
        r'(?:GIVEN\s+NAMES?)\s*:?\s*([A-Z\s]+?)(?:\s+DATE|$)',
    ]
    
    for i, pattern in enumerate(name_patterns):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if i == 0:
                data["surname"] = match.group(1).strip()
            else:
                data["given_names"] = match.group(1).strip()
    
    # Country patterns
    country_patterns = [
        r'(?:COUNTRY\s+CODE|ISSUING\s+COUNTRY)\s*:?\s*([A-Z]{3})',
        r'(?:NATIONALITY)\s*:?\s*([A-Z]{3})',
        r'PAKISTAN',  # Specific country detection
        r'INDIA',
        r'USA',
    ]
    
    for pattern in country_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            country_text = match.group(0).upper()
            if 'PAKISTAN' in country_text:
                data["country_code"] = 'PAK'
            elif 'INDIA' in country_text:
                data["country_code"] = 'IND'
            elif 'USA' in country_text:
                data["country_code"] = 'USA'
            else:
                data["country_code"] = match.group(1)
            break
    
    # Date patterns (more flexible for Tesseract)
    date_patterns = [
        (r'(?:DATE\s+OF\s+BIRTH|DOB)\s*:?\s*(\d{2}[/\-]\d{2}[/\-]\d{4})', "date_of_birth"),
        (r'(?:EXPIRY|EXPIRATION)\s*:?\s*(\d{2}[/\-]\d{2}[/\-]\d{4})', "expiry_date"),
        (r'(\d{2}\s+[A-Z]{3}\s+\d{4})', "date_of_birth"),  # DD MMM YYYY format
    ]
    
    for pattern, field in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            # Convert to YYMMDD format
            if '/' in date_str or '-' in date_str:
                parts = re.split(r'[/\-]', date_str)
                if len(parts) == 3:
                    day, month, year = parts
                    if len(year) == 4:
                        year = year[2:]  # Take last 2 digits
                    data[field] = f"{year}{month.zfill(2)}{day.zfill(2)}"
            elif ' ' in date_str:  # DD MMM YYYY format
                parts = date_str.split()
                if len(parts) == 3:
                    day, month_name, year = parts
                    month_map = {
                        'JAN':'01','FEB':'02','MAR':'03','APR':'04','MAY':'05','JUN':'06',
                        'JUL':'07','AUG':'08','SEP':'09','OCT':'10','NOV':'11','DEC':'12'
                    }
                    month = month_map.get(month_name.upper()[:3], '01')
                    if len(year) == 4:
                        year = year[2:]
                    data[field] = f"{year}{month}{day.zfill(2)}"
    
    # Sex pattern
    sex_match = re.search(r'(?:SEX|GENDER)\s*:?\s*([MFX])', text, re.IGNORECASE)
    if sex_match:
        data["sex"] = sex_match.group(1).upper()
    
    if verbose:
        print(f"    Extracted data: {data}")
    
    return data


def reconstruct_mrz_from_data(data: Dict, verbose: bool = False) -> str:
    """
    Reconstruct MRZ from extracted passport data
    (Same as EasyOCR version)
    """
    try:
        # Required fields
        country = data.get("country_code", "XXX")[:3]
        surname = data.get("surname", "UNKNOWN")[:20]
        given_names = data.get("given_names", "UNKNOWN")[:15]
        passport_num = data.get("passport_number", "000000000")[:9]
        nationality = data.get("nationality", country)[:3]
        dob = data.get("date_of_birth", "000000")[:6]
        sex = data.get("sex", "<")[:1]
        expiry = data.get("expiry_date", "000000")[:6]
        
        # Build Line 1: P<CCCSSSSSSSSSSSS<<GGGGGGGGGGGGGGGGGGG
        name_field = f"{surname}<<{given_names.replace(' ', '<')}"
        if len(name_field) > 39:
            name_field = name_field[:39]
        name_field = name_field.ljust(39, '<')
        line1 = f"P<{country}{name_field}"
        
        # Build Line 2: TD3 format
        passport_field = passport_num.ljust(9, '<')
        line2 = f"{passport_field}<{nationality}{dob}<{sex}{expiry}<<<<<<<<<<<<<<<"
        
        # Ensure exactly 44 characters
        line1 = line1[:44].ljust(44, '<')
        line2 = line2[:44].ljust(44, '<')
        
        mrz_text = f"{line1}\n{line2}"
        
        if verbose:
            print(f"    Reconstructed MRZ:")
            print(f"      Line 1: {line1}")
            print(f"      Line 2: {line2}")
        
        return mrz_text
    
    except Exception as e:
        if verbose:
            print(f"    ✗ MRZ reconstruction failed: {e}")
        return ""