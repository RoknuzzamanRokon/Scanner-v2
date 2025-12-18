"""
Super Optimized Tesseract OCR fallback validation with enhanced text extraction (FIXED VERSION)
Builds on extraction_focus version with additional improvements for better results
"""
import cv2
import numpy as np
import re
import tempfile
import os
from PIL import Image
from typing import Dict, Optional, Tuple
import pytesseract
from country_code import get_country_info
from utils import format_mrz_date

# Enhanced cache for previous validation failures
_validation_failure_cache = {}

# Enhanced cache for country information
_country_info_cache = {}

# Cache for image analysis results to avoid redundant calculations
_image_analysis_cache = {}

def validate_passport_with_tesseract_fallback(image: Image.Image, verbose: bool = True, user_id: str = None) -> Dict:
    """
    Super Optimized Tesseract OCR Fallback Validation
    
    Enhanced Features:
    - Advanced image quality analysis and enhancement
    - Multi-stage OCR with progressive quality improvement
    - Enhanced MRZ extraction with better pattern recognition
    - Adaptive preprocessing based on image characteristics
    - Comprehensive error recovery and fallback strategies
    
    Args:
        image: PIL Image object
        verbose: Print detailed logs
        
    Returns:
        Dictionary with success status and passport data
    """
    try:
        if verbose:
            print(f"  -> Processing with Super Optimized Tesseract OCR...")
        
        # Set Tesseract path from environment (cached)
        from config import config
        if hasattr(config, 'TESSERACT_CMD') and config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
            if verbose:
                print(f"  -> Using Tesseract: {config.TESSERACT_CMD}")
        
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if verbose:
            print(f"  -> Applying advanced image analysis and preprocessing...")
        
        # Enhanced preprocessing with image quality analysis
        processed_images = enhanced_preprocess_for_passport_ocr(img_cv, verbose)
        
        if verbose:
            print(f"  -> Generated {len(processed_images)} enhanced preprocessed variants")
        
        best_result = None
        best_confidence = 0
        
        # Enhanced OCR configurations with additional optimizations
        ocr_configs = get_enhanced_ocr_configs()
        
        # Multi-stage OCR with progressive quality improvement
        best_result, best_confidence = multi_stage_ocr_attempts(
            processed_images, ocr_configs, img_cv, verbose
        )
        
        if not best_result:
            if verbose:
                print(f"  X Tesseract: No meaningful text detected in any variant")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "method_used": "TesseractOCR",
                "error": "No meaningful text detected by Tesseract"
            }
        
        extracted_text = best_result
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            if verbose:
                print(f"  X Tesseract: No meaningful text detected")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "method_used": "TesseractOCR",
                "error": "No meaningful text detected by Tesseract"
            }
        
        # Enhanced text cleaning and processing (FIXED)
        lines, all_text = fixed_enhanced_text_processing(extracted_text, verbose)
        
        # Advanced passport indicator check with more patterns
        passport_score = advanced_passport_indicators(all_text, lines)
        
        if verbose:
            print(f"  -> Passport indicators found: {passport_score}/14")
        
        if passport_score < 3:  # Lower threshold for initial check
            if verbose:
                print(f"  X No passport indicators found in text")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "method_used": "TesseractOCR",
                "error": "No passport indicators found in extracted text"
            }
        
        # Enhanced MRZ extraction with multiple strategies
        mrz_lines = enhanced_mrz_extraction(lines, all_text, verbose)
        
        if not mrz_lines:
            if verbose:
                print(f"  X Could not extract or reconstruct valid MRZ")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "method_used": "TesseractOCR",
                "error": "Could not extract or reconstruct valid MRZ from text"
            }
        
        # Validate MRZ using TD3 validation checker
        mrz_text = '\n'.join(mrz_lines)
        
        if verbose:
            print(f"  -> Validating MRZ with TD3 rules...")
            print(f"    Line 1: {mrz_lines[0]}")
            print(f"    Line 2: {mrz_lines[1]}")
        
        from passport_detector import passport_validation_checker
        
        validation_result = passport_validation_checker(mrz_text, verbose=False)
        is_valid = validation_result.get("is_valid", False)
        confidence = validation_result.get("confidence_score", 0.0)
        
        if verbose:
            print(f"    TD3 Valid: {is_valid}")
            print(f"    Confidence: {confidence*100:.1f}%")
        
        # Enhanced validation with lower confidence threshold
        if not is_valid or confidence < 0.4:  # Lower threshold for super optimized
            if verbose:
                print(f"  X MRZ validation failed: {validation_result.get('reason', 'Low confidence')}")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": mrz_text,
                "method_used": "TesseractOCR",
                "error": f"MRZ validation failed: {validation_result.get('reason', 'Low confidence')} (confidence: {confidence*100:.1f}%)"
            }
        
        # Extract passport data from validated MRZ
        passport_data = validation_result.get("passport_data", {})
        
        # Enhanced country information with additional caching
        country_code = passport_data.get("country_code", "")
        if country_code:
            country_info = get_enhanced_country_info(country_code)
            passport_data.update({
                "country_name": country_info.get("name", country_code),
                "country_iso": country_info.get("alpha2", ""),
                "nationality": country_info.get("nationality", country_code)
            })
        
        # Enhanced date formatting with additional validation
        if passport_data.get("date_of_birth"):
            passport_data["date_of_birth"] = enhanced_format_date(passport_data["date_of_birth"])
        if passport_data.get("expiry_date"):
            passport_data["expiry_date"] = enhanced_format_date(passport_data["expiry_date"])
        
        # Enhanced field validation with more flexible threshold
        from utils import check_field_validation_threshold
        validation_check = check_field_validation_threshold(mrz_text, threshold=8, verbose=verbose)  # Lower threshold
        
        if not validation_check["threshold_met"]:
            if verbose:
                print(f"!! Field validation threshold not met: {validation_check['valid_count']}/8 fields valid")
                print(f"   -> Proceeding to next validation method...")
            
            # Enhanced validation failure handling
            if user_id:
                from utils import save_validation_failure
                save_validation_failure(user_id, "TesseractOCR", passport_data, validation_check["field_results"], mrz_text, all_text)
            
            return {
                "success": False,
                "passport_data": passport_data,
                "mrz_text": mrz_text,
                "method_used": "TesseractOCR",
                "error": f"Field validation threshold not met: {validation_check['valid_count']}/8 fields valid",
                "validation_summary": validation_check
            }
        
        if verbose:
            print(f"OK Field validation threshold met: {validation_check['valid_count']}/8 fields valid")
            print(f"   -> Returning validated passport data...")
            print(f"  OK Passport data extracted successfully")
            print(f"    Surname: {passport_data.get('surname', '')}")
            print(f"    Given Names: {passport_data.get('given_names', '')}")
            print(f"    Passport #: {passport_data.get('passport_number', '')}")
            print(f"    Country: {passport_data.get('country_name', '')} ({country_code})")
        
        return {
            "success": True,
            "passport_data": passport_data,
            "mrz_text": mrz_text,
            "method_used": "TesseractOCR",
            "error": "",
            "validation_summary": validation_check
        }
    
    except Exception as e:
        import traceback
        if verbose:
            print(f"  X Tesseract error: {e}")
            traceback.print_exc()
        
        return {
            "success": False,
            "passport_data": {},
            "mrz_text": "",
            "method_used": "TesseractOCR",
            "error": f"Tesseract processing error: {str(e)}"
        }

def fixed_enhanced_text_processing(text: str, verbose: bool = False) -> Tuple[list, str]:
    """
    FIXED Enhanced text cleaning and processing
    
    Args:
        text: Raw OCR text
        verbose: Print debug info
        
    Returns:
        Tuple of (cleaned_lines, all_text)
    """
    # Basic cleaning
    lines = text.strip().split('\n')
    
    # Enhanced line cleaning (FIXED)
    cleaned_lines = []
    for line in lines:
        # Remove excessive whitespace
        clean_line = ' '.join(line.split())
        
        # Remove common OCR artifacts (FIXED - removed problematic regex)
        clean_line = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', clean_line)  # Remove control characters
        # Use safer pattern for removing special characters
        clean_line = re.sub(r'[^\w\s<>-]', '', clean_line)  # Keep word chars, spaces, <>, and hyphens
        
        if clean_line.strip():
            cleaned_lines.append(clean_line)
    
    # Enhanced all text processing
    all_text = ' '.join(cleaned_lines)
    
    # Additional cleaning for common OCR errors (FIXED)
    all_text = re.sub(r'[\\/:*?"<>|]', ' ', all_text)  # Remove problematic characters
    all_text = re.sub(r'\s+', ' ', all_text)  # Normalize whitespace
    
    return cleaned_lines, all_text

# Copy all other functions from the super_optimized version
# (They are working correctly, just need the fixed text processing)

def enhanced_preprocess_for_passport_ocr(img: np.ndarray, verbose: bool = False) -> list:
    """
    Enhanced preprocessing with advanced image quality analysis
    
    Args:
        img: OpenCV image (BGR format)
        verbose: Print debug info
        
    Returns:
        List of preprocessed images
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Advanced image quality analysis
    height, width = gray.shape
    aspect_ratio = width / height
    
    # Calculate enhanced image quality metrics
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Additional quality metrics
    entropy = cv2.calcHist([gray], [0], None, [256], [0, 256])
    entropy = -np.sum([p * np.log2(p) for p in entropy if p > 0])
    
    if verbose:
        print(f"    Enhanced image analysis: {width}x{height}, AR={aspect_ratio:.2f}")
        print(f"      Blur: {blur_score:.1f}, Brightness: {mean_brightness:.1f}")
        print(f"      Contrast: {contrast:.1f}, Entropy: {entropy:.1f}")
    
    processed_images = []
    
    # Always include enhanced preprocessing (most reliable)
    processed_images.append(preprocess_for_passport_ocr(img, method="enhanced"))
    
    # Add MRZ-focused preprocessing if image has good contrast
    if contrast > 25:  # Lower threshold for more aggressive MRZ processing
        processed_images.append(preprocess_for_passport_ocr(img, method="mrz_focused"))
    
    # Add direct method for fast processing (low overhead)
    processed_images.append(preprocess_for_passport_ocr(img, method="direct"))
    
    # Add high contrast method if image is too dark or bright
    if mean_brightness < 100 or mean_brightness > 200:
        processed_images.append(preprocess_for_passport_ocr(img, method="high_contrast"))
    
    # Only add MRZ region detection if image is large enough
    if width > 500 and height > 300:  # Lower threshold for MRZ detection
        mrz_region = detect_mrz_region(img)
        if mrz_region is not None:
            processed_images.append(preprocess_for_passport_ocr(mrz_region, method="mrz_only"))
    
    return processed_images

def get_enhanced_ocr_configs() -> list:
    """
    Return enhanced OCR configurations with additional optimizations
    
    Returns:
        List of OCR configuration strings
    """
    return [
        # PSM 6: Single uniform block (most effective for passports)
        r'--psm 6',
        # MRZ-specific config with enhanced character whitelist
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<>',
        # Direct OCR without special config
        r'',
        # Single text line mode for MRZ with enhanced whitelist
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<>',
        # Additional: PSM 11 for sparse text (MRZ-like)
        r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<>',
    ]

def multi_stage_ocr_attempts(processed_images: list, ocr_configs: list, original_img: np.ndarray, verbose: bool = False) -> Tuple[Optional[str], float]:
    """
    Multi-stage OCR attempts with progressive quality improvement
    
    Args:
        processed_images: List of preprocessed images
        ocr_configs: List of OCR configurations
        original_img: Original image for fallback
        verbose: Print debug info
        
    Returns:
        Tuple of (best_result, best_confidence)
    """
    best_result = None
    best_confidence = 0
    
    # Stage 1: Try all processed images with all configs
    for i, processed_img in enumerate(processed_images):
        for j, config in enumerate(ocr_configs):
            try:
                if verbose:
                    print(f"    -> Stage 1 - Trying OCR variant {i+1}.{j+1}...")
                
                # Early exit optimization
                if best_confidence > 0.90:  # Higher threshold for early exit
                    if verbose:
                        print(f"      >> Skipping remaining configs (excellent result: {best_confidence:.2f})")
                    break
                
                import time
                start_time = time.time()
                
                # No timeout - let OCR run to completion
                try:
                    extracted_text = pytesseract.image_to_string(processed_img, config=config)
                except Exception as e:
                    if verbose:
                        print(f"      X OCR variant {i+1}.{j+1} failed: {e}")
                    continue
                
                processing_time = time.time() - start_time
                
                if verbose:
                    print(f"      >> Took {processing_time:.1f}s")
                
                if not extracted_text or len(extracted_text.strip()) < 10:
                    continue
                
                # Enhanced quality evaluation
                text_quality = enhanced_evaluate_ocr_quality(extracted_text, verbose=False)
                
                if text_quality > best_confidence:
                    best_confidence = text_quality
                    best_result = extracted_text
                    if verbose:
                        print(f"      OK New best result (quality: {text_quality:.2f})")
                
                # Early termination for excellent results
                if text_quality > 0.90:
                    if verbose:
                        print(f"      OK Excellent result found, stopping search")
                    return best_result, best_confidence
                
            except Exception as e:
                if verbose:
                    print(f"      X OCR variant {i+1}.{j+1} failed: {e}")
                continue
        
        # Break outer loop if excellent quality found
        if best_confidence > 0.90:
            break
    
    # Stage 2: Try additional enhancement techniques if results are mediocre
    if best_confidence < 0.70:
        if verbose:
            print(f"    -> Stage 2 - Applying additional enhancement techniques...")
        
        # Try image sharpening
        try:
            sharpened = apply_image_sharpening(original_img)
            enhanced_text = pytesseract.image_to_string(sharpened, config='--psm 6')
            if enhanced_text and len(enhanced_text.strip()) > 15:
                enhanced_quality = enhanced_evaluate_ocr_quality(enhanced_text, verbose=False)
                if enhanced_quality > best_confidence:
                    best_confidence = enhanced_quality
                    best_result = enhanced_text
                    if verbose:
                        print(f"      OK Image sharpening improved result (quality: {enhanced_quality:.2f})")
        except Exception:
            pass
        
        # Try contrast enhancement
        try:
            contrasted = apply_contrast_enhancement(original_img)
            contrasted_text = pytesseract.image_to_string(contrasted, config='--psm 6')
            if contrasted_text and len(contrasted_text.strip()) > 15:
                contrasted_quality = enhanced_evaluate_ocr_quality(contrasted_text, verbose=False)
                if contrasted_quality > best_confidence:
                    best_confidence = contrasted_quality
                    best_result = contrasted_text
                    if verbose:
                        print(f"      OK Contrast enhancement improved result (quality: {contrasted_quality:.2f})")
        except Exception:
            pass
    
    # Stage 3: Fallback to simple OCR if still no good results
    if not best_result or best_confidence < 0.60:
        if verbose:
            print(f"    -> Stage 3 - Trying comprehensive fallback strategies...")
        
        # Try multiple fallback configurations
        fallback_configs = [
            '--psm 6',
            '--psm 7',
            '--psm 11',
            '--psm 12',
            '--oem 3 --psm 6'
        ]
        
        for config in fallback_configs:
            try:
                fallback_text = pytesseract.image_to_string(original_img, config=config)
                if fallback_text and len(fallback_text.strip()) > 15:
                    fallback_quality = enhanced_evaluate_ocr_quality(fallback_text, verbose=False)
                    if fallback_quality > best_confidence:
                        best_confidence = fallback_quality
                        best_result = fallback_text
                        if verbose:
                            print(f"      OK Fallback OCR succeeded (quality: {fallback_quality:.2f})")
                        break
            except Exception:
                continue
    
    return best_result, best_confidence

def advanced_passport_indicators(all_text: str, lines: list) -> int:
    """
    Advanced passport indicator check with more patterns
    
    Args:
        all_text: All extracted text
        lines: List of text lines
        
    Returns:
        Passport indicator score
    """
    passport_indicators = [
        "PASSPORT" in all_text.upper(),
        "REPUBLIC" in all_text.upper(),
        "PEOPLE" in all_text.upper(),
        "BANGLADESH" in all_text.upper(),
        "INDIA" in all_text.upper(),
        "PAKISTAN" in all_text.upper(),
        "USA" in all_text.upper(),
        "UNITED" in all_text.upper(),
        "STATE" in all_text.upper(),
        bool(re.search(r"<{2,}", all_text)),  # MRZ padding
        bool(re.search(r"P<[A-Z]{3}", all_text)),  # MRZ passport type
        bool(re.search(r"[A-Z0-9]{8,}", all_text)),  # Long alphanumeric
        bool(re.search(r"\d{6}", all_text)),  # Date patterns
        len([line for line in lines if len(line.strip()) > 30]) >= 2,  # Long lines
    ]
    
    return sum(passport_indicators)

def enhanced_mrz_extraction(lines: list, all_text: str, verbose: bool = False) -> Optional[list]:
    """
    Enhanced MRZ extraction with multiple strategies
    
    Args:
        lines: List of text lines
        all_text: All extracted text
        verbose: Print debug info
        
    Returns:
        List of MRZ lines or None
    """
    # Strategy 1: Try direct MRZ line extraction (enhanced)
    mrz_lines = enhanced_direct_mrz_extraction(lines, verbose)
    if mrz_lines:
        return mrz_lines
    
    # Strategy 2: Try advanced MRZ pattern extraction
    mrz_lines = advanced_mrz_pattern_extraction(all_text, verbose)
    if mrz_lines:
        return mrz_lines
    
    # Strategy 3: Try enhanced data extraction and reconstruction
    mrz_lines = enhanced_data_reconstruction(all_text, verbose)
    if mrz_lines:
        return mrz_lines
    
    # Strategy 4: Try MRZ reconstruction from partial data
    mrz_lines = partial_mrz_reconstruction(all_text, verbose)
    if mrz_lines:
        return mrz_lines
    
    return None

def enhanced_direct_mrz_extraction(lines: list, verbose: bool = False) -> Optional[list]:
    """
    Enhanced direct MRZ line extraction
    
    Args:
        lines: List of text lines
        verbose: Print debug info
        
    Returns:
        List of MRZ lines or None
    """
    potential_mrz = []
    
    for line in lines:
        clean_line = line.strip().upper()
        
        # Enhanced MRZ criteria with more patterns
        mrz_criteria = [
            '<' in clean_line and len(clean_line) >= 18,  # More lenient length
            len(clean_line) >= 30 and bool(re.search(r'[A-Z0-9]{8,}', clean_line)),
            clean_line.startswith('P') and len(clean_line) >= 25,
            bool(re.search(r'\d{6,}', clean_line)) and len(clean_line) >= 20,
            len(clean_line) >= 35,  # Lower threshold
            bool(re.search(r'[A-Z]{3}[A-Z0-9]{5,}', clean_line)),  # Country + number pattern
        ]
        
        if any(mrz_criteria):
            potential_mrz.append(clean_line)
    
    if len(potential_mrz) >= 2:
        # Enhanced line selection with better heuristics
        line1_candidate = None
        line2_candidate = None
        
        # Look for line starting with 'P' for line 1
        for mrz in potential_mrz:
            if mrz.startswith('P') and len(mrz) >= 25:
                line1_candidate = mrz
                break
        
        if not line1_candidate:
            # Try to find line with country code pattern
            for mrz in potential_mrz:
                if re.search(r'P<[A-Z]{3}', mrz):
                    line1_candidate = mrz
                    break
        
        if not line1_candidate:
            return None
        
        # Find best line 2 candidate with enhanced scoring
        remaining_lines = [mrz for mrz in potential_mrz if mrz != line1_candidate]
        if not remaining_lines:
            return None
        
        # Score remaining candidates
        best_score = -1
        for line in remaining_lines:
            score = 0
            if '<' in line:
                score += line.count('<') * 3
            if re.search(r'\d{6}', line):
                score += 15
            if len(line) >= 40:
                score += 10
            if score > best_score:
                best_score = score
                line2_candidate = line
        
        # Enhanced cleaning and formatting
        line1 = enhanced_clean_mrz_line(line1_candidate)
        line2 = enhanced_clean_mrz_line(line2_candidate)
        
        # More flexible padding
        if len(line1) >= 30:
            line1 = line1[:44].ljust(44, '<')
        if len(line2) >= 30:
            line2 = line2[:44].ljust(44, '<')
        
        if len(line1) == 44 and len(line2) == 44:
            return [line1, line2]
        elif len(line1) >= 40 and len(line2) >= 40:
            # Accept slightly shorter lines
            return [line1[:44].ljust(44, '<'), line2[:44].ljust(44, '<')]
    
    return None

def advanced_mrz_pattern_extraction(text: str, verbose: bool = False) -> Optional[list]:
    """
    Advanced MRZ pattern extraction with better scoring
    
    Args:
        text: Raw OCR text
        verbose: Print debug info
        
    Returns:
        List of MRZ lines or None
    """
    lines = text.strip().split('\n')
    potential_mrz = []
    
    for line in lines:
        clean_line = line.strip().upper()
        
        if len(clean_line) < 18:  # More lenient minimum length
            continue
        
        # Enhanced scoring with more criteria
        score = 0
        
        # Length scoring
        if 40 <= len(clean_line) <= 50:
            score += 15
        elif 30 <= len(clean_line) <= 60:
            score += 10
        elif 20 <= len(clean_line) < 30:
            score += 5
        
        # Character pattern scoring
        if re.search(r'P[A-Z]{3}', clean_line):  # P + country code
            score += 20
        
        if re.search(r'[A-Z0-9]{8,}', clean_line):  # Long alphanumeric
            score += 12
        
        if re.search(r'\d{6}', clean_line):  # Date patterns
            score += 15
        
        if '<' in clean_line:
            score += clean_line.count('<') * 3
        
        # Position scoring
        if re.search(r'[A-Z]{3}[A-Z0-9]{5,}', clean_line):  # Country + number
            score += 10
        
        if score >= 12:  # Lower threshold
            potential_mrz.append((score, clean_line))
    
    # Sort by score and get top candidates
    potential_mrz.sort(key=lambda x: x[0], reverse=True)
    
    if len(potential_mrz) >= 2:
        # Try to form MRZ lines with enhanced matching
        line1_candidate = None
        line2_candidate = None
        
        # Look for passport type line (starts with P)
        for score, candidate in potential_mrz:
            if candidate.startswith('P') and len(candidate) >= 25:
                line1_candidate = candidate
                break
        
        if not line1_candidate:
            # Try to find line with country code pattern
            for score, candidate in potential_mrz:
                if re.search(r'P<[A-Z]{3}', candidate):
                    line1_candidate = candidate
                    break
        
        if not line1_candidate:
            return None
        
        # Find best second line
        remaining = [candidate for score, candidate in potential_mrz if candidate != line1_candidate]
        if not remaining:
            return None
        
        # Use the highest scoring remaining candidate
        line2_candidate = remaining[0]
        
        # Enhanced cleaning and formatting
        line1 = enhanced_clean_mrz_line(line1_candidate)
        line2 = enhanced_clean_mrz_line(line2_candidate)
        
        # More flexible padding
        if len(line1) >= 30:
            line1 = line1[:44].ljust(44, '<')
        if len(line2) >= 30:
            line2 = line2[:44].ljust(44, '<')
        
        if len(line1) == 44 and len(line2) == 44:
            return [line1, line2]
        elif len(line1) >= 40 and len(line2) >= 40:
            return [line1[:44].ljust(44, '<'), line2[:44].ljust(44, '<')]
    
    return None

def enhanced_data_reconstruction(text: str, verbose: bool = False) -> Optional[list]:
    """
    Enhanced data extraction and MRZ reconstruction
    
    Args:
        text: Raw OCR text
        verbose: Print debug info
        
    Returns:
        List of MRZ lines or None
    """
    # Enhanced passport data extraction
    passport_data = enhanced_extract_passport_data_from_text(text, verbose)
    
    if not passport_data:
        return None
    
    # Try to reconstruct MRZ from extracted data
    reconstructed_mrz = enhanced_reconstruct_mrz_from_data(passport_data, verbose)
    
    if reconstructed_mrz:
        mrz_lines = reconstructed_mrz.split('\n')
        return mrz_lines
    
    return None

def partial_mrz_reconstruction(text: str, verbose: bool = False) -> Optional[list]:
    """
    Try to reconstruct MRZ from partial data patterns
    
    Args:
        text: Raw OCR text
        verbose: Print debug info
        
    Returns:
        List of MRZ lines or None
    """
    # Look for partial MRZ patterns and try to reconstruct
    lines = text.strip().split('\n')
    
    # Try to find any lines that look like MRZ fragments
    mrz_fragments = []
    for line in lines:
        clean_line = line.strip().upper()
        if len(clean_line) >= 20 and ('<' in clean_line or re.search(r'[A-Z]{3}[A-Z0-9]{5,}', clean_line)):
            mrz_fragments.append(clean_line)
    
    if len(mrz_fragments) >= 2:
        # Try to combine fragments into valid MRZ lines
        line1_candidate = None
        line2_candidate = None
        
        # Look for line with passport type indicator
        for frag in mrz_fragments:
            if frag.startswith('P') or 'P<' in frag:
                line1_candidate = frag
                break
        
        if not line1_candidate:
            line1_candidate = mrz_fragments[0]
        
        # Find the best second line candidate
        remaining = [frag for frag in mrz_fragments if frag != line1_candidate]
        if remaining:
            line2_candidate = remaining[0]
        else:
            line2_candidate = mrz_fragments[1] if len(mrz_fragments) > 1 else line1_candidate
        
        # Try to clean and format
        line1 = enhanced_clean_mrz_line(line1_candidate)
        line2 = enhanced_clean_mrz_line(line2_candidate)
        
        # More flexible formatting
        if len(line1) >= 30:
            line1 = line1[:44].ljust(44, '<')
        if len(line2) >= 30:
            line2 = line2[:44].ljust(44, '<')
        
        if len(line1) >= 40 and len(line2) >= 40:
            return [line1, line2]
    
    return None

def enhanced_clean_mrz_line(line: str) -> str:
    """
    Enhanced MRZ line cleaning with better error correction
    
    Args:
        line: Raw MRZ line text
        
    Returns:
        Cleaned MRZ line
    """
    # Basic cleaning
    cleaned = line.replace(' ', '').upper()
    
    # Enhanced character replacements with more patterns
    replacements = {
        '0': 'O',  # Zero to O
        '1': 'I',  # One to I
        '8': 'B',  # Eight to B
        '5': 'S',  # Five to S
        '6': 'G',  # Six to G
        '2': 'Z',  # Two to Z
        '7': 'T',  # Seven to T
        '9': 'G',  # Nine to G
        '4': 'A',  # Four to A
        '3': 'E',  # Three to E
    }
    
    # Apply replacements more intelligently
    if cleaned.startswith('P<') and len(cleaned) > 5:
        # Only apply in name section for line 1
        for old, new in replacements.items():
            if old in cleaned[5:]:
                cleaned = cleaned[:5] + cleaned[5:].replace(old, new)
    else:
        # For line 2, be more conservative with replacements
        name_section_end = min(len(cleaned), 15)  # Only replace in name part
        for old, new in replacements.items():
            if old in cleaned[:name_section_end]:
                cleaned = cleaned.replace(old, new, 1)  # Only replace first occurrence
    
    return cleaned

def enhanced_evaluate_ocr_quality(text: str, verbose: bool = False) -> float:
    """
    Enhanced OCR quality evaluation with more factors
    
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
    
    # Factor 1: Presence of passport-related keywords (enhanced)
    passport_keywords = ['PASSPORT', 'REPUBLIC', 'PEOPLE', 'BANGLADESH', 'INDIA', 'USA', 'PAKISTAN', 'UNITED', 'STATE', 'GOVERNMENT']
    keyword_count = sum(1 for keyword in passport_keywords if keyword in text.upper())
    keyword_score = min(keyword_count * 0.15, 0.4)  # More weight to keywords
    score += keyword_score
    factors.append(f"Keywords: {keyword_score:.2f}")
    
    # Factor 2: Presence of MRZ-like patterns (enhanced)
    mrz_patterns = [
        r'P<[A-Z]{3}',  # Passport type and country
        r'<{2,}',       # MRZ padding
        r'[A-Z0-9]{6,}', # Long alphanumeric sequences
        r'\d{6}',       # Date patterns
        r'[A-Z]{3}[A-Z0-9]{5,}', # Country + number
    ]
    
    mrz_score = 0
    for pattern in mrz_patterns:
        if re.search(pattern, text):
            mrz_score += 0.1
    
    mrz_score = min(mrz_score, 0.35)  # More weight to MRZ patterns
    score += mrz_score
    factors.append(f"MRZ patterns: {mrz_score:.2f}")
    
    # Factor 3: Text readability (enhanced)
    total_chars = len(text)
    readable_chars = len(re.findall(r'[A-Za-z0-9<>\s]', text))
    readability = readable_chars / total_chars if total_chars > 0 else 0
    readability_score = readability * 0.25  # More weight to readability
    score += readability_score
    factors.append(f"Readability: {readability_score:.2f}")
    
    # Factor 4: Penalize excessive garbage characters (enhanced)
    garbage_chars = len(re.findall(r'[^\w\s<>]', text))
    garbage_penalty = min(garbage_chars / total_chars * 0.25, 0.25) if total_chars > 0 else 0
    score -= garbage_penalty
    factors.append(f"Garbage penalty: -{garbage_penalty:.2f}")
    
    # Factor 5: Bonus for structured text (enhanced)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) >= 2:
        structure_bonus = 0.15
        score += structure_bonus
        factors.append(f"Structure bonus: {structure_bonus:.2f}")
    
    # Factor 6: Bonus for MRZ-like line structure
    mrz_lines = [line for line in lines if 30 <= len(line) <= 50]
    if len(mrz_lines) >= 2:
        mrz_structure_bonus = 0.1
        score += mrz_structure_bonus
        factors.append(f"MRZ structure bonus: {mrz_structure_bonus:.2f}")
    
    # Normalize score
    final_score = max(0.0, min(1.0, score))
    
    if verbose:
        print(f"    Enhanced OCR Quality Evaluation: {final_score:.2f}")
        for factor in factors:
            print(f"      {factor}")
    
    return final_score

def get_enhanced_country_info(country_code: str) -> Dict:
    """
    Enhanced country information with additional caching
    
    Args:
        country_code: Country code
        
    Returns:
        Country information dictionary
    """
    if country_code not in _country_info_cache:
        country_info = get_country_info(country_code)
        # Enhance with additional country data if available
        if country_info:
            _country_info_cache[country_code] = country_info
        else:
            # Fallback to basic country code mapping
            _country_info_cache[country_code] = {
                "name": country_code,
                "alpha2": country_code[:2] if len(country_code) >= 2 else country_code,
                "nationality": country_code + " citizen"
            }
    
    return _country_info_cache[country_code]

def enhanced_format_date(date_str: str) -> str:
    """
    Enhanced date formatting with additional validation
    
    Args:
        date_str: Date string to format
        
    Returns:
        Formatted date string
    """
    if not date_str or len(date_str) != 6:
        return date_str
    
    try:
        # Basic validation
        if not date_str.isdigit():
            return date_str
        
        # Extract components
        year = date_str[:2]
        month = date_str[2:4]
        day = date_str[4:6]
        
        # Basic validation
        if int(month) < 1 or int(month) > 12:
            return date_str
        if int(day) < 1 or int(day) > 31:
            return date_str
        
        # Return formatted date
        return f"{year}-{month}-{day}"
    
    except Exception:
        return date_str

def apply_image_sharpening(img: np.ndarray) -> np.ndarray:
    """
    Apply image sharpening to enhance text
    
    Args:
        img: Input image
        
    Returns:
        Sharpened image
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    return sharpened

def apply_contrast_enhancement(img: np.ndarray) -> np.ndarray:
    """
    Apply contrast enhancement to improve text visibility
    
    Args:
        img: Input image
        
    Returns:
        Contrast-enhanced image
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced

# Copy all remaining helper functions from the previous version
# (preprocess_for_passport_ocr, detect_mrz_region, enhanced_extract_passport_data_from_text, enhanced_reconstruct_mrz_from_data)

def preprocess_for_passport_ocr(img: np.ndarray, method: str = "enhanced") -> np.ndarray:
    """Existing preprocessing function"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    if method == "enhanced":
        height, width = gray.shape
        gray = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        gray = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        return binary
    
    elif method == "mrz_focused":
        height, width = gray.shape
        gray = cv2.resize(gray, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel_sharpen)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary
    
    elif method == "direct":
        return gray
    
    elif method == "mrz_only":
        height, width = gray.shape
        gray = cv2.resize(gray, (width * 5, height * 5), interpolation=cv2.INTER_CUBIC)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary
    
    elif method == "high_contrast":
        height, width = gray.shape
        gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        gray = cv2.equalizeHist(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.dilate(binary, kernel, iterations=1)
        return binary
    
    else:
        gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2))
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary

def detect_mrz_region(img: np.ndarray) -> Optional[np.ndarray]:
    """Existing MRZ region detection function"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape
        potential_mrz_regions = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (y > height * 0.6 and w > width * 0.4 and h > 20 and h < height * 0.15 and w / h > 5):
                potential_mrz_regions.append((x, y, w, h, w * h))
        
        if not potential_mrz_regions:
            return None
        
        potential_mrz_regions.sort(key=lambda x: x[4], reverse=True)
        x, y, w, h, _ = potential_mrz_regions[0]
        
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        mrz_region = gray[y:y+h, x:x+w]
        return mrz_region
    
    except Exception:
        return None

def enhanced_extract_passport_data_from_text(text: str, verbose: bool = False) -> Dict:
    """Enhanced passport data extraction function"""
    data = {}
    
    # Enhanced passport number patterns
    passport_patterns = [
        r'(?:PASSPORT\s+(?:NO|NUMBER|#)\.?\s*:?\s*)([A-Z0-9]+)',
        r'(?:DOCUMENT\s+(?:NO|NUMBER|#)\.?\s*:?\s*)([A-Z0-9]+)',
        r'(?:NO\.?\s*:?\s*)([A-Z0-9]{6,12})',
        r'([A-Z]{2}\d{7})',
        r'([A-Z]\d{7})',
        r'(\d{8,})',
    ]
    
    for pattern in passport_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["passport_number"] = match.group(1)
            break
    
    # Enhanced name patterns
    name_patterns = [
        r'(?:SURNAME|FAMILY\s+NAME)\s*:?\s*([A-Z\s]+?)(?:\s+GIVEN|$)',
        r'(?:GIVEN\s+NAMES?)\s*:?\s*([A-Z\s]+?)(?:\s+DATE|$)',
        r'(?:NAME)\s*:?\s*([A-Z\s]+?)(?:\s+PASSPORT|$)',
    ]
    
    for i, pattern in enumerate(name_patterns):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if i == 0:
                data["surname"] = match.group(1).strip()
            else:
                data["given_names"] = match.group(1).strip()
    
    # Enhanced country patterns
    country_patterns = [
        r'(?:COUNTRY\s+CODE|ISSUING\s+COUNTRY)\s*:?\s*([A-Z]{3})',
        r'(?:NATIONALITY)\s*:?\s*([A-Z]{3})',
        r'PAKISTAN',
        r'INDIA',
        r'USA',
        r'BANGLADESH',
        r'UNITED\s+STATES',
    ]
    
    for pattern in country_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            country_text = match.group(0).upper()
            if 'PAKISTAN' in country_text:
                data["country_code"] = 'PAK'
            elif 'INDIA' in country_text:
                data["country_code"] = 'IND'
            elif 'USA' in country_text or 'UNITED STATES' in country_text:
                data["country_code"] = 'USA'
            elif 'BANGLADESH' in country_text:
                data["country_code"] = 'BGD'
            else:
                data["country_code"] = match.group(1)
            break
    
    # Enhanced date patterns
    date_patterns = [
        (r'(?:DATE\s+OF\s+BIRTH|DOB)\s*:?\s*(\d{2}[/\-]\d{2}[/\-]\d{4})', "date_of_birth"),
        (r'(?:EXPIRY|EXPIRATION)\s*:?\s*(\d{2}[/\-]\d{2}[/\-]\d{4})', "expiry_date"),
        (r'(\d{2}\s+[A-Z]{3}\s+\d{4})', "date_of_birth"),
        (r'(\d{6})', "date_of_birth"),  # YYMMDD format
    ]
    
    for pattern, field in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            if '/' in date_str or '-' in date_str:
                parts = re.split(r'[/\-]', date_str)
                if len(parts) == 3:
                    day, month, year = parts
                    if len(year) == 4:
                        year = year[2:]
                    data[field] = f"{year}{month.zfill(2)}{day.zfill(2)}"
            elif ' ' in date_str:
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
            else:
                # Handle YYMMDD format
                if len(date_str) == 6 and date_str.isdigit():
                    data[field] = date_str
    
    # Enhanced sex pattern with proper error handling
    sex_patterns = [
        r'(?:SEX|GENDER)\s*:?\s*([MFX])',
        r'(?:M|F|X)\s*(?:ALE|EMALE)?',
    ]
    
    for pattern in sex_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                from sex_field_normalizer import normalize_sex_field
                # Check if group exists before accessing
                if match.lastindex and match.lastindex >= 1:
                    sex_text = match.group(1).upper()
                else:
                    sex_text = match.group(0).upper()
                
                if sex_text in ['M', 'F', 'X']:
                    data["sex"] = normalize_sex_field(sex_text)
                elif 'MALE' in sex_text:
                    data["sex"] = 'M'
                elif 'FEMALE' in sex_text:
                    data["sex"] = 'F'
                break
            except Exception as e:
                if verbose:
                    print(f"      Warning: Sex field extraction failed: {e}")
                continue
    
    return data

def enhanced_reconstruct_mrz_from_data(data: Dict, verbose: bool = False) -> str:
    """Enhanced MRZ reconstruction function"""
    try:
        country = data.get("country_code", "")[:3]
        surname = data.get("surname", "")[:20]
        given_names = data.get("given_names", "")[:15]
        passport_num = data.get("passport_number", "")[:9]
        nationality = data.get("nationality", country)[:3]
        dob = data.get("date_of_birth", "")[:6]
        from sex_field_normalizer import normalize_sex_field
        sex = normalize_sex_field(data.get("sex", "<"))[:1]
        expiry = data.get("expiry_date", "")[:6]
        
        # Enhanced name field construction
        name_field = f"{surname}<<{given_names.replace(' ', '<')}"
        if len(name_field) > 39:
            name_field = name_field[:39]
        name_field = name_field.ljust(39, '<')
        line1 = f"P<{country}{name_field}"
        
        # Enhanced passport field construction
        passport_field = passport_num.ljust(9, '<')
        line2 = f"{passport_field}<{nationality}{dob}<{sex}{expiry}<<<<<<<<<<<<<<<"
        
        # Ensure exactly 44 characters
        line1 = line1[:44].ljust(44, '<')
        line2 = line2[:44].ljust(44, '<')
        
        mrz_text = f"{line1}\n{line2}"
        
        return mrz_text
    
    except Exception:
        return ""