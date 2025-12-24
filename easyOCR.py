"""
EasyOCR fallback validation with MRZ reconstruction and TD3 compliance
"""
import easyocr
import cv2
import numpy as np
import re
import tempfile
import os
from PIL import Image
from typing import Dict
from country_code import get_country_info
from utils import format_mrz_date


def validate_passport_with_easyocr_fallback(image: Image.Image, verbose: bool = True, user_folder: str = None, user_id: str = None) -> Dict:
    """
    STEP 3: EasyOCR Fallback Validation
    
    - EasyOCR text extraction
    - MRZ pattern detection and reconstruction
    - TD3 format validation & cleaning
    - Meaningful data validation
    
    Args:
        image: PIL Image object
        verbose: Print detailed logs
        
    Returns:
        Dictionary with success status and passport data
    """
    try:
        if verbose:
            print(f"  → Processing with EasyOCR...")
        
        # Check for previous validation failures
        if user_id:
            from utils import load_validation_failures, analyze_previous_failures
            previous_failures = load_validation_failures(user_id)
            if previous_failures:
                analysis = analyze_previous_failures(previous_failures, "EasyOCR")
                if analysis["suggestions"]:
                    print(f"💡 Suggestions based on previous failures:")
                    for suggestion in analysis["suggestions"]:
                        print(f"   → {suggestion}")
        
        # Handle EXIF orientation to match OpenCV loading behavior
        # PIL automatically applies EXIF rotation, but OpenCV doesn't
        # We need to "undo" PIL's automatic rotation to match OpenCV behavior
        
        try:
            exif = image.getexif()
            orientation = exif.get(274) if exif else None  # 274 is EXIF orientation tag
            
            if verbose:
                print(f"    Debug: EXIF orientation: {orientation}")
            
            # Undo PIL's automatic EXIF rotation to match OpenCV behavior
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(-90, expand=True)  # Undo the 90° CW rotation
            elif orientation == 8:
                image = image.rotate(90, expand=True)   # Undo the 90° CCW rotation
                
        except Exception as e:
            if verbose:
                print(f"    Debug: Could not process EXIF: {e}")
        
        # Convert PIL to OpenCV format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and then to OpenCV format
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale exactly like standalone script
        processed_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        if verbose:
            print(f"  → Running EasyOCR text extraction...")
        
        # Initialize EasyOCR reader with exact same settings as working version
        reader = easyocr.Reader(['en'], gpu=False)  # Remove verbose=False to match standalone
        
        # Save debug image to user-specific temp folder
        debug_path = None
        if verbose:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_filename = f"debug_easyocr_{timestamp}.jpg"
            
            if user_folder:
                # Save to user-specific temp folder
                debug_path = os.path.join(user_folder, debug_filename)
            else:
                # Fallback to temp folder
                debug_path = os.path.join("temp", debug_filename)
                
            cv2.imwrite(debug_path, processed_image)
            print(f"    Processed image shape: {processed_image.shape}")
            print(f"    Debug image saved: {debug_path}")
        
        # Extract text - use detail=0 like the working version for simpler text extraction
        results_text = reader.readtext(processed_image, detail=0)
        
        if verbose:
            print(f"    EasyOCR extracted {len(results_text)} text segments")
            
        # Also get detailed results for confidence checking if needed
        results_detailed = []
        if len(results_text) < 10:  # Only get detailed if we have few results
            results_detailed = reader.readtext(processed_image, detail=1)
        
        if not results_text:
            if verbose:
                print(f"  ✗ EasyOCR: No text detected")
            
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "method_used": "EasyOCR",
                "error": "No text detected by EasyOCR"
            }
        
        # Extract all text - use the simple text results like the working version
        all_text = [text.upper() for text in results_text]
        full_text = " ".join(all_text)
        
        # Also get high confidence text for MRZ detection
        high_confidence_text = []
        for (bbox, text, confidence) in results_detailed:
            if confidence > 0.5:  # Filter low confidence text for MRZ detection
                high_confidence_text.append(text.upper())
        
        if verbose:
            print(f"  ✓ Text extracted: {len(all_text)} segments")
            print(f"    Full text preview: {full_text[:200]}...")
            print(f"    All extracted segments:")
            for i, text in enumerate(all_text):
                print(f"      {i+1}: '{text}'")
        
        # Check if this looks like a passport - use same logic as working version
        is_passport_word = "PASSPORT" in full_text
        has_mrz_chars = bool(re.search(r"<{5,}", full_text))  # Look for 5+ consecutive < like working version
        
        if verbose:
            print(f"    Passport word found: {is_passport_word}")
            print(f"    MRZ pattern found: {has_mrz_chars}")
        
        # Check passport validation
        if is_passport_word or has_mrz_chars:
            if verbose:
                print(f"  ✓ Valid for passport image")
        else:
            if verbose:
                print(f"  ✗ Document doesn't appear to be a passport")
            
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "method_used": "EasyOCR",
                "error": "Document doesn't appear to be a passport (no passport keywords or MRZ patterns)"
            }
        
        # Try to extract MRZ lines
        mrz_lines = []
        potential_mrz = []
        
        # Look for lines that could be MRZ (contain < symbols and are long)
        # Use both all_text and high_confidence_text for MRZ detection
        combined_text = list(set(all_text + high_confidence_text))
        for text in combined_text:
            if '<' in text and len(text) >= 20:
                potential_mrz.append(text)
        
        if verbose:
            print(f"  → Found {len(potential_mrz)} potential MRZ segments")
            for i, mrz in enumerate(potential_mrz):
                print(f"    MRZ {i+1}: {mrz}")
                print(f"      Starts with 'P': {mrz.startswith('P')}")
                print(f"      Length: {len(mrz)}")
                print(f"      Pattern match: {bool(re.match(r'^[A-Z0-9]', mrz))}")
        
        # Try to reconstruct MRZ from potential segments
        if len(potential_mrz) >= 2:
            # Find the correct order for TD3 format
            # Line 1 should start with 'P' (passport type)
            # Line 2 should contain passport number and other data
            
            line1_candidate = None
            line2_candidate = None
            line1_score = 0
            line2_score = 0
            
            # FIRST: Look for line starting with 'P' - this MUST be Line 1 in TD3 format
            for mrz in potential_mrz:
                if mrz.startswith('P') and len(mrz) >= 20:  # Changed from 'P<' to 'P' to be more flexible
                    # Score Line 1 candidate
                    score = 100  # Perfect start
                    if len(mrz) == 44:
                        score = 100  # Perfect length
                    elif 40 <= len(mrz) <= 50:
                        score = 90   # Good length
                    elif 25 <= len(mrz) <= 39:
                        score = 80   # Acceptable length (may need padding)
                    else:
                        score = 70   # Short but usable
                    
                    # Check for proper TD3 Line 1 format: P<CCC...
                    if re.match(r'^P<[A-Z]{3}', mrz):
                        score = 100
                    elif mrz.startswith('P') and '<' in mrz:  # More flexible pattern matching
                        score = 90
                    
                    if score > line1_score:
                        line1_score = score
                        line1_candidate = mrz
            
            # SECOND: Find the second line from remaining candidates
            # TD3 Line 2 has specific characteristics: passport number + check digits + dates
            # IMPORTANT: Exclude the Line 1 candidate from consideration
            remaining_lines = [mrz for mrz in potential_mrz if mrz != line1_candidate]
            if remaining_lines:
                for line in remaining_lines:
                    score = 0
                    
                    # Score based on TD3 Line 2 characteristics
                    # Should have pattern: NNNNNNNNN<CCCDDDDDD<SDDDDDD<PPPPPPPPPPPPP<C<C
                    # Where N=passport number, C=country, D=dates, S=sex, P=personal number
                    
                    # Check for proper TD3 Line 2 patterns
                    if re.match(r'^[A-Z0-9]{8,9}[0-9<][A-Z]{3}[0-9]{6}[0-9<][MFX<][0-9]{6}', line):
                        score = 100  # Perfect TD3 Line 2 format
                    elif re.search(r'[A-Z0-9]{6,}[A-Z]{3}[0-9]{6}[MFX][0-9]{6}', line):
                        score = 90   # Good TD3-like pattern
                    elif re.search(r'[A-Z0-9]{4,}.*[A-Z]{3}.*[0-9]{6}.*[MFX].*[0-9]{6}', line):
                        score = 80   # Recognizable TD3 elements
                    else:
                        # Fallback scoring for less perfect lines
                        if '<' in line:
                            score += line.count('<') * 2  # More < symbols = more likely MRZ
                        
                        # Check for numeric patterns (dates, passport numbers)
                        if re.search(r'\d{6,}', line):  # 6+ consecutive digits (dates)
                            score += 20
                        
                        # Check for typical MRZ length or close to it
                        if 40 <= len(line) <= 50:
                            score += 10
                        elif len(line) >= 35:
                            score += 5
                        
                        # Penalize lines with common words (not MRZ)
                        if any(word in line.upper() for word in ['REPUBLIC', 'PEOPLE', 'BANGLADESH', 'PASSPORT', 'MINISTRY']):
                            score -= 30
                        
                        # CRITICAL: Penalize lines that start with 'P' - they should be Line 1, not Line 2
                        if line.startswith('P'):
                            score = 0  # Force score to 0 - this cannot be Line 2
                    
                    if verbose:
                        print(f"      Line '{line[:30]}...' score: {score}")
                    
                    if score > line2_score:
                        line2_score = score
                        line2_candidate = line
            
            # THIRD: Fallback logic - ensure TD3 format compliance
            if not line1_candidate or not line2_candidate:
                # Sort by length but maintain TD3 format rules
                potential_mrz.sort(key=len, reverse=True)
                
                # Find any line starting with 'P' for Line 1
                for mrz in potential_mrz:
                    if mrz.startswith('P') and not line1_candidate:
                        line1_candidate = mrz
                        line1_score = 50  # Fallback score
                        break
                
                # Find any remaining line for Line 2
                for mrz in potential_mrz:
                    if mrz != line1_candidate and not line2_candidate:
                        line2_candidate = mrz
                        line2_score = 50  # Fallback score
                        break
            
            # FINAL VALIDATION: Ensure TD3 format compliance
            # Line 1 MUST start with 'P', Line 2 MUST NOT start with 'P'
            if line1_candidate and line2_candidate:
                if not line1_candidate.startswith('P') and line2_candidate.startswith('P'):
                    # Swap them - Line 2 should be Line 1
                    if verbose:
                        print(f"    → Swapping lines to maintain TD3 format (Line 1 must start with 'P')")
                    line1_candidate, line2_candidate = line2_candidate, line1_candidate
                    line1_score, line2_score = line2_score, line1_score
            
            if verbose:
                print(f"    Selected Line 1: {line1_candidate}")
                print(f"    Selected Line 1 score: {line1_score}")
                print(f"    Selected Line 2: {line2_candidate}")
                print(f"    Selected Line 2 score: {line2_score}")
            
            # Decision logic: Use direct lines if scores are high enough
            if line1_score >= 90 and line2_score >= 80:
                # High confidence - use lines directly with minimal cleaning
                if verbose:
                    print(f"  ✓ High confidence MRZ lines found - using directly")
                
                line1 = clean_mrz_line(line1_candidate)
                line2 = clean_mrz_line(line2_candidate)
                
                # Ensure proper length (pad to 44 characters)
                if len(line1) >= 25:  # More lenient minimum for Line 1 (names can be short)
                    line1 = line1[:44].ljust(44, '<')
                if len(line2) >= 35:  # More lenient minimum for Line 2
                    line2 = line2[:44].ljust(44, '<')
                
                # Accept if both lines are now 44 characters and basic validation passes
                if len(line1) == 44 and len(line2) == 44 and line1.startswith('P'):
                    mrz_lines = [line1, line2]
                    if verbose:
                        print(f"  ✓ Direct MRZ lines validated and padded to 44 characters")
                        print(f"    Final Line 1: {line1}")
                        print(f"    Final Line 2: {line2}")
                else:
                    if verbose:
                        print(f"  ⚠ Direct MRZ lines failed validation (L1:{len(line1)}, L2:{len(line2)})")
                    
                    # Try TD3 MRZ reconstruction when direct lines fail validation
                    if verbose:
                        print(f"  → Attempting TD3 MRZ reconstruction...")
                    
                    # Extract passport data from full text for reconstruction
                    passport_data = extract_passport_data_from_text(full_text, verbose)
                    
                    # Reconstruct MRZ using TD3 format rules
                    reconstructed_mrz = reconstruct_mrz_from_candidates_and_data(
                        line1_candidate, line2_candidate, passport_data, verbose
                    )
                    
                    if reconstructed_mrz:
                        mrz_lines = reconstructed_mrz.split('\n')
                        if verbose:
                            print(f"  ✓ MRZ reconstructed from extracted data")
                            print(f"    Reconstructed Line 1: {mrz_lines[0]}")
                            print(f"    Reconstructed Line 2: {mrz_lines[1]}")
                    else:
                        if verbose:
                            print(f"  ⚠ TD3 reconstruction failed, trying medium confidence path")
            
            elif line1_score >= 60 or line2_score >= 60:
                # Medium confidence - try to fix/reconstruct
                if verbose:
                    print(f"  → Medium confidence - attempting MRZ reconstruction")
                
                # Try to use what we have and reconstruct missing parts
                line1 = clean_mrz_line(line1_candidate) if line1_candidate else ""
                line2 = clean_mrz_line(line2_candidate) if line2_candidate else ""
                
                # If Line 1 is good but Line 2 needs work
                if line1_score >= 90 and line2_score < 80:
                    if verbose:
                        print(f"    → Line 1 good, reconstructing Line 2")
                    
                    # Extract data from text and reconstruct Line 2
                    passport_data = extract_passport_data_from_text(full_text, verbose)
                    if passport_data:
                        reconstructed_line2 = reconstruct_td3_line2_from_data(passport_data, line2_candidate, verbose)
                        if reconstructed_line2:
                            line2 = reconstructed_line2
                
                # Ensure proper length
                if len(line1) >= 40:
                    line1 = line1[:44].ljust(44, '<')
                if len(line2) >= 40:
                    line2 = line2[:44].ljust(44, '<')
                
                if len(line1) == 44 and len(line2) == 44:
                    mrz_lines = [line1, line2]
        
        # If we couldn't get proper MRZ lines, try to extract data from full text
        if not mrz_lines:
            if verbose:
                print(f"  → No valid MRZ lines found, attempting data extraction from full text")
            
            # Extract passport data from full text
            passport_data = extract_passport_data_from_text(full_text, verbose)
            
            if not passport_data.get("passport_number") and not passport_data.get("surname"):
                if verbose:
                    print(f"  ✗ Could not extract meaningful passport data")
                
                return {
                    "success": False,
                    "passport_data": {},
                    "mrz_text": "",
                    "method_used": "EasyOCR",
                    "error": "Could not extract meaningful passport data from text"
                }
            
            # Try to reconstruct MRZ from extracted data
            reconstructed_mrz = reconstruct_mrz_from_data(passport_data, verbose)
            
            if reconstructed_mrz:
                mrz_lines = reconstructed_mrz.split('\n')
                if verbose:
                    print(f"  ✓ MRZ reconstructed from extracted data")
            else:
                if verbose:
                    print(f"  ⚠ Could not reconstruct valid MRZ")
                
                return {
                    "success": False,
                    "passport_data": passport_data,
                    "mrz_text": "",
                    "method_used": "EasyOCR",
                    "error": "Could not reconstruct valid MRZ from extracted data"
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
                "method_used": "EasyOCR",
                "error": f"MRZ validation failed: {validation_result.get('reason', 'Low confidence')} (confidence: {confidence*100:.1f}%)"
            }
            
        
        mrz_for_validation =  f"{mrz_lines[0]}" + "\n" + f"{mrz_lines[1]}"  
            
        # Import and use the passport field validation function
        from passport_check import validate_passport_fields
        
        field_results = validate_passport_fields(mrz_for_validation)
        # Display results in terminal
        for field, status in field_results.items():
            status_icon = "✅" if status == "Valid" else "❌"
            print(f"{status_icon} {field:20}: {status}")
        
        
        # Summary
        valid_count = sum(1 for status in field_results.values() if status == "Valid")
        total_count = len(field_results)
        print(f"\nField Validation Summary: {valid_count}/{total_count} fields are valid\n\n")
        
            
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
        
        # Check field validation threshold
        from utils import check_field_validation_threshold
        validation_check = check_field_validation_threshold(mrz_text, threshold=10, verbose=verbose)
        
        if not validation_check["threshold_met"]:
            if verbose:
                print(f"⚠️  Field validation threshold not met: {validation_check['valid_count']}/10 fields valid")
                print(f"   → Proceeding to next validation method...")
            
            # Save validation failure to temp file for user
            if user_id:
                from utils import save_validation_failure
                save_validation_failure(user_id, "EasyOCR", passport_data, validation_check["field_results"], mrz_text, full_text)
            
            return {
                "success": False,
                "passport_data": passport_data,
                "mrz_text": mrz_text,
                "method_used": "EasyOCR",
                "error": f"Field validation threshold not met: {validation_check['valid_count']}/10 fields valid",
                "validation_summary": validation_check
            }
        
        if verbose:
            print(f"✅ Field validation threshold met: {validation_check['valid_count']}/10 fields valid")
            print(f"   → Returning validated passport data...")
            print(f"  ✓ Passport data extracted successfully")
            print(f"    Surname: {passport_data.get('surname', '')}")
            print(f"    Given Names: {passport_data.get('given_names', '')}")
            print(f"    Passport #: {passport_data.get('passport_number', '')}")
            print(f"    Country: {passport_data.get('country_name', '')} ({country_code})")
        
        # Note: User folder cleanup will be handled by scanner.py on success
        
        return {
            "success": True,
            "passport_data": passport_data,
            "mrz_text": mrz_text,
            "method_used": "EasyOCR",
            "error": "",
            "validation_summary": validation_check
        }
    
    except Exception as e:
        import traceback
        if verbose:
            print(f"  ✗ EasyOCR error: {e}")
            traceback.print_exc()
        
        return {
            "success": False,
            "passport_data": {},
            "mrz_text": "",
            "method_used": "EasyOCR",
            "error": f"EasyOCR processing error: {str(e)}"
        }
    



def clean_mrz_line(line: str) -> str:
    """
    Clean and normalize MRZ line
    
    Args:
        line: Raw MRZ line text
        
    Returns:
        Cleaned MRZ line
    """
    # Remove spaces and convert to uppercase
    cleaned = line.replace(' ', '').upper()
    
    # Replace common OCR errors
    replacements = {
        '0': 'O',  # Zero to O in names
        '1': 'I',  # One to I in names
        '8': 'B',  # Eight to B in names (sometimes)
        '5': 'S',  # Five to S in names (sometimes)
    }
    
    # Only apply replacements in name sections (not in dates/numbers)
    if cleaned.startswith('P<'):
        # This is line 1 (names), apply letter replacements
        for old, new in replacements.items():
            if old in cleaned[5:]:  # Only in name section
                cleaned = cleaned[:5] + cleaned[5:].replace(old, new)
    
    return cleaned


def extract_passport_data_from_text(text: str, verbose: bool = False) -> Dict:
    """
    Extract passport data from full text using pattern matching
    
    Args:
        text: Full extracted text
        verbose: Print detailed logs
        
    Returns:
        Dictionary with extracted passport data
    """
    data = {}
    
    # Passport number patterns
    passport_patterns = [
        r'(?:PASSPORT\s+(?:NO|NUMBER|#)\.?\s*:?\s*)([A-Z0-9]+)',
        r'(?:DOCUMENT\s+(?:NO|NUMBER|#)\.?\s*:?\s*)([A-Z0-9]+)',
        r'(?:NO\.?\s*:?\s*)([A-Z0-9]{6,12})',
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
    ]
    
    for pattern in country_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["country_code"] = match.group(1)
            break
    
    # Date patterns
    date_patterns = [
        (r'(?:DATE\s+OF\s+BIRTH|DOB)\s*:?\s*(\d{2}[/\-]\d{2}[/\-]\d{4})', "date_of_birth"),
        (r'(?:EXPIRY|EXPIRATION)\s*:?\s*(\d{2}[/\-]\d{2}[/\-]\d{4})', "expiry_date"),
    ]
    
    for pattern, field in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            # Convert to YYMMDD format
            parts = re.split(r'[/\-]', date_str)
            if len(parts) == 3:
                day, month, year = parts
                if len(year) == 4:
                    year = year[2:]  # Take last 2 digits
                data[field] = f"{year}{month.zfill(2)}{day.zfill(2)}"
    
    # Sex pattern
    sex_match = re.search(r'(?:SEX|GENDER)\s*:?\s*([MFX])', text, re.IGNORECASE)
    if sex_match:
        from sex_field_normalizer import normalize_sex_field
        data["sex"] = normalize_sex_field(sex_match.group(1).upper())
    
    if verbose:
        print(f"    Extracted data: {data}")
    
    return data


def reconstruct_td3_line2_from_data(data: Dict, original_line2: str, verbose: bool = False) -> str:
    """
    Reconstruct TD3 Line 2 from extracted data and original line
    
    Args:
        data: Extracted passport data
        original_line2: Original Line 2 candidate
        verbose: Print detailed logs
        
    Returns:
        Reconstructed TD3 Line 2 (44 chars) or empty string if failed
    """
    try:
        # Extract what we can from the original line
        passport_num = ""
        nationality = ""
        dob = ""
        sex = ""
        expiry = ""
        
        # Try to extract from original line first
        if original_line2:
            # Look for patterns in the original line
            # Passport number (usually at start)
            passport_match = re.match(r'^([A-Z0-9]{6,9})', original_line2)
            if passport_match:
                passport_num = passport_match.group(1)
            
            # Country code (3 letters)
            country_match = re.search(r'([A-Z]{3})', original_line2)
            if country_match:
                nationality = country_match.group(1)
            
            # Dates (6 digits)
            date_matches = re.findall(r'(\d{6})', original_line2)
            if len(date_matches) >= 2:
                dob = date_matches[0]
                expiry = date_matches[1]
            elif len(date_matches) == 1:
                # Try to determine if it's DOB or expiry based on year
                year = int(date_matches[0][:2])
                if year > 50:  # Likely birth year (19xx)
                    dob = date_matches[0]
                else:  # Likely expiry year (20xx)
                    expiry = date_matches[0]
            
            # Sex
            sex_match = re.search(r'([MFX])', original_line2)
            if sex_match:
                sex = sex_match.group(1)
        
        # Fill in missing data from extracted data
        if not passport_num and data.get("passport_number"):
            passport_num = data["passport_number"][:9]
        if not nationality and data.get("nationality"):
            nationality = data["nationality"][:3]
        if not dob and data.get("date_of_birth"):
            dob = data["date_of_birth"][:6]
        if not sex and data.get("sex"):
            from sex_field_normalizer import normalize_sex_field
            sex = normalize_sex_field(data["sex"])[:1]
        if not expiry and data.get("expiry_date"):
            expiry = data["expiry_date"][:6]
        
        # Build TD3 Line 2: NNNNNNNNN<CCCDDDDDD<SDDDDDD<PPPPPPPPPPPPP<C<C
        passport_field = passport_num.ljust(9, '<')[:9]
        
        # Calculate check digit for passport number
        passport_check = calculate_mrz_check_digit(passport_num)
        
        nationality_field = nationality.ljust(3, '<')[:3]
        dob_field = dob.ljust(6, '0')[:6] if dob else '000000'
        
        # Calculate check digit for DOB
        dob_check = calculate_mrz_check_digit(dob_field)
        
        sex_field = sex if sex in 'MFX' else '<'
        expiry_field = expiry.ljust(6, '0')[:6] if expiry else '000000'
        
        # Calculate check digit for expiry
        expiry_check = calculate_mrz_check_digit(expiry_field)
        
        # Personal number field (usually empty, filled with <)
        personal_field = '<' * 14
        personal_check = '<'
        
        # Build line 2
        line2_parts = [
            passport_field,      # 9 chars: passport number
            passport_check,      # 1 char: passport check digit
            nationality_field,   # 3 chars: nationality
            dob_field,          # 6 chars: date of birth
            dob_check,          # 1 char: DOB check digit
            sex_field,          # 1 char: sex
            expiry_field,       # 6 chars: expiry date
            expiry_check,       # 1 char: expiry check digit
            personal_field,     # 14 chars: personal number
            personal_check,     # 1 char: personal number check
        ]
        
        line2_partial = ''.join(line2_parts)  # 43 chars so far
        
        # Calculate final check digit for the entire line 2
        final_check = calculate_mrz_check_digit(line2_partial)
        
        line2 = line2_partial + final_check  # 44 chars total
        
        if verbose:
            print(f"    → Reconstructed TD3 Line 2: {line2}")
        
        return line2
        
    except Exception as e:
        if verbose:
            print(f"    ✗ TD3 Line 2 reconstruction failed: {e}")
        return ""


def calculate_mrz_check_digit(data: str) -> str:
    """
    Calculate MRZ check digit using TD3 algorithm
    
    Args:
        data: Data string to calculate check digit for
        
    Returns:
        Single character check digit (0-9)
    """
    weights = [7, 3, 1]
    total = 0
    
    for i, char in enumerate(data):
        if char.isdigit():
            value = int(char)
        elif char == '<':
            value = 0
        elif char.isalpha():
            # A=10, B=11, ..., Z=35
            value = ord(char.upper()) - ord('A') + 10
        else:
            value = 0
        
        total += value * weights[i % 3]
    
    return str(total % 10)


def reconstruct_mrz_from_candidates_and_data(line1_candidate: str, line2_candidate: str, passport_data: Dict, verbose: bool = False) -> str:
    """
    Reconstruct MRZ from candidate lines and extracted data using TD3 format rules
    
    Args:
        line1_candidate: Original Line 1 candidate (may be short)
        line2_candidate: Original Line 2 candidate 
        passport_data: Extracted passport data from full text
        verbose: Print detailed logs
        
    Returns:
        Reconstructed MRZ text (44 chars per line) or empty string if failed
    """
    try:
        # Extract data from Line 1 candidate (name information)
        surname = ""
        given_names = ""
        
        if line1_candidate and line1_candidate.startswith('P'):
            # Parse Line 1: P<CCCNAME_FIELD
            # Remove P< prefix and country code to get name field
            name_part = line1_candidate[5:] if len(line1_candidate) > 5 else line1_candidate[1:]
            
            # Split by << to separate surname and given names
            if '<<' in name_part:
                parts = name_part.split('<<', 1)
                surname = parts[0].strip('<')
                given_names = parts[1].strip('<').replace('<', ' ').strip() if len(parts) > 1 else ""
            else:
                # Fallback: treat as surname only
                surname = name_part.strip('<')
        
        # Extract data from Line 2 candidate (passport number, dates, etc.)
        passport_number = ""
        nationality = ""
        birth_date = ""
        sex = ""
        expiry_date = ""
        personal_number = ""
        
        if line2_candidate:
            # Try to extract passport number (usually at start)
            passport_match = re.match(r'^([A-Z0-9]{6,9})', line2_candidate)
            if passport_match:
                passport_number = passport_match.group(1)
            
            # Extract country code (3 letters) - look for it after passport number
            country_match = re.search(r'([A-Z]{3})', line2_candidate[9:])  # Skip passport number area
            if country_match:
                nationality = country_match.group(1)
            
            # For TD3 format, dates are at specific positions if line is 44 chars
            if len(line2_candidate) == 44:
                # TD3 Line 2 format: PASSPORT(9)CHECK(1)COUNTRY(3)BIRTH(6)CHECK(1)SEX(1)EXPIRY(6)CHECK(1)PERSONAL(14)CHECK(1)FINAL(1)
                birth_date = line2_candidate[13:19]  # Position 13-18
                sex = line2_candidate[20:21]         # Position 20
                expiry_date = line2_candidate[21:27] # Position 21-26
                personal_number = line2_candidate[28:42].rstrip('<')  # Position 28-41
            else:
                # Fallback: extract dates (6 digits each) in order
                date_matches = re.findall(r'(\d{6})', line2_candidate)
                if len(date_matches) >= 2:
                    birth_date = date_matches[0]
                    expiry_date = date_matches[1]
                
                # Extract sex
                sex_match = re.search(r'([MFX])', line2_candidate)
                if sex_match:
                    sex = sex_match.group(1)
                
                # Extract personal number (remaining digits after dates)
                personal_match = re.search(r'(\d{10,})', line2_candidate)
                if personal_match:
                    personal_number = personal_match.group(1)
        
        # Fill missing data from extracted passport data
        if not surname and passport_data.get("surname"):
            surname = passport_data["surname"]
        if not given_names and passport_data.get("given_names"):
            given_names = passport_data["given_names"]
        if not passport_number and passport_data.get("passport_number"):
            passport_number = passport_data["passport_number"]
        if not nationality and passport_data.get("nationality"):
            nationality = passport_data["nationality"]
        if not birth_date and passport_data.get("date_of_birth"):
            # Convert from YYYY-MM-DD to YYMMDD
            dob = passport_data["date_of_birth"]
            if len(dob) == 10 and '-' in dob:  # YYYY-MM-DD format
                parts = dob.split('-')
                birth_date = parts[0][2:] + parts[1] + parts[2]  # YYMMDD
        if not sex and passport_data.get("sex"):
            from sex_field_normalizer import normalize_sex_field
            sex = normalize_sex_field(passport_data["sex"])
        if not expiry_date and passport_data.get("expiry_date"):
            # Convert from YYYY-MM-DD to YYMMDD
            exp = passport_data["expiry_date"]
            if len(exp) == 10 and '-' in exp:  # YYYY-MM-DD format
                parts = exp.split('-')
                expiry_date = parts[0][2:] + parts[1] + parts[2]  # YYMMDD
        
        # Use default country if not found
        if not nationality:
            nationality = "PAK"  # Default based on user's case
        
        if verbose:
            print(f"    Extracted/Combined data:")
            print(f"      Surname: {surname}")
            print(f"      Given Names: {given_names}")
            print(f"      Passport: {passport_number}")
            print(f"      Nationality: {nationality}")
            print(f"      Birth Date: {birth_date}")
            print(f"      Sex: {sex}")
            print(f"      Expiry: {expiry_date}")
        
        # Validate we have minimum required data
        if not surname or not passport_number:
            if verbose:
                print(f"    ✗ Missing critical data (surname: {bool(surname)}, passport: {bool(passport_number)})")
            return ""
        
        # Reconstruct TD3 Line 1: P<CCCNAME_FIELD (44 chars)
        document_type = "P"
        country_code = nationality[:3] if nationality else "XXX"
        
        # Format name field: SURNAME<<GIVEN<NAMES
        name_field = surname.upper()
        if given_names:
            name_field += "<<" + given_names.upper().replace(' ', '<')
        else:
            name_field += "<<"  # Empty given names
        
        # Build Line 1
        line1_prefix = f"{document_type}<{country_code}"
        available_space = 44 - len(line1_prefix)
        name_field_padded = name_field[:available_space].ljust(available_space, '<')
        line1 = line1_prefix + name_field_padded
        
        # Reconstruct TD3 Line 2: PASSPORT_NUM<CHECK<COUNTRY<BIRTH<CHECK<SEX<EXPIRY<CHECK<PERSONAL<CHECK<FINAL_CHECK (44 chars)
        passport_field = passport_number[:9].ljust(9, '<')
        passport_check = calculate_mrz_check_digit(passport_number)
        
        nationality_field = nationality[:3].ljust(3, '<')
        birth_field = birth_date[:6].ljust(6, '0') if birth_date else '000000'
        birth_check = calculate_mrz_check_digit(birth_field)
        
        sex_field = sex if sex in 'MFX' else '<'
        expiry_field = expiry_date[:6].ljust(6, '0') if expiry_date else '000000'
        expiry_check = calculate_mrz_check_digit(expiry_field)
        
        # Personal number field (14 chars)
        if personal_number:
            personal_field = personal_number[:14].ljust(14, '<')
        else:
            personal_field = '<' * 14
        personal_check = calculate_mrz_check_digit(personal_field.rstrip('<'))
        
        # Build Line 2 (43 chars so far)
        line2_partial = (passport_field + passport_check + nationality_field + 
                        birth_field + birth_check + sex_field + expiry_field + 
                        expiry_check + personal_field + personal_check)
        
        # Calculate final check digit
        final_check = calculate_mrz_check_digit(line2_partial)
        line2 = line2_partial + final_check
        
        # Ensure both lines are exactly 44 characters
        line1 = line1[:44].ljust(44, '<')
        line2 = line2[:44].ljust(44, '<')
        
        if verbose:
            print(f"    Reconstructed MRZ:")
            print(f"      Line 1: {line1}")
            print(f"      Line 2: {line2}")
        
        return f"{line1}\n{line2}"
        
    except Exception as e:
        if verbose:
            print(f"    ✗ MRZ reconstruction error: {e}")
        return ""


def reconstruct_mrz_from_data(data: Dict, verbose: bool = False) -> str:
    """
    Reconstruct MRZ from extracted passport data
    
    Args:
        data: Extracted passport data
        verbose: Print detailed logs
        
    Returns:
        Reconstructed MRZ text (2 lines, 44 chars each) or empty string if failed
    """
    try:
        # Required fields
        country = data.get("country_code", "")[:3]
        surname = data.get("surname", "")[:20]
        given_names = data.get("given_names", "")[:15]
        passport_num = data.get("passport_number", "")[:9]
        nationality = data.get("nationality", country)[:3]
        dob = data.get("date_of_birth", "")[:6]
        from sex_field_normalizer import normalize_sex_field
        sex = normalize_sex_field(data.get("sex", "<"))[:1]
        expiry = data.get("expiry_date", "")[:6]
        
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
