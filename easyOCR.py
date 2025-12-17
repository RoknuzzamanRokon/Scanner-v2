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
        
        if not (is_passport_word or has_mrz_chars):
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
            
            if verbose:
                print(f"    Selected Line 1 candidate: {line1_candidate}")
                print(f"    Selected Line 2 candidate: {line2_candidate}")
            
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
