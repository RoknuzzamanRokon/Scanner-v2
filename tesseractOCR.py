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
    Tesseract OCR Fallback Validation
    
    - Tesseract OCR text extraction
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
            print(f"  → Processing with Tesseract OCR...")
        
        # Set Tesseract path from environment
        from config import config
        if hasattr(config, 'TESSERACT_CMD') and config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
            if verbose:
                print(f"  → Using Tesseract: {config.TESSERACT_CMD}")
        
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocessing for better OCR
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing for better text recognition
        # 1. Upscale the image
        gray = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2))
        
        # 2. Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 3. Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        if verbose:
            print(f"  → Running Tesseract OCR text extraction...")
        
        # Extract text using Tesseract with specific config for passport documents
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
        
        try:
            # Try with custom config first
            extracted_text = pytesseract.image_to_string(thresh, config=custom_config)
        except:
            # Fallback to default config
            extracted_text = pytesseract.image_to_string(thresh)
        
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
        
        # Check if this looks like a passport
        is_passport_word = "PASSPORT" in all_text.upper()
        has_mrz_chars = bool(re.search(r"<{3,}", all_text))  # Look for MRZ patterns
        
        if not (is_passport_word or has_mrz_chars):
            if verbose:
                print(f"  ✗ Document doesn't appear to be a passport")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "method_used": "Tesseract",
                "error": "Document doesn't appear to be a passport (no passport keywords or MRZ patterns)"
            }
        
        # Try to extract MRZ lines
        mrz_lines = []
        potential_mrz = []
        
        # Look for lines that could be MRZ (contain < symbols and are long)
        for line in lines:
            clean_line = line.strip().upper()
            if '<' in clean_line and len(clean_line) >= 20:
                potential_mrz.append(clean_line)
        
        if verbose:
            print(f"  → Found {len(potential_mrz)} potential MRZ segments")
            for i, mrz in enumerate(potential_mrz):
                print(f"    MRZ {i+1}: {mrz}")
        
        # Try to reconstruct MRZ from potential segments
        if len(potential_mrz) >= 2:
            # Sort by length (longer lines first)
            potential_mrz.sort(key=len, reverse=True)
            
            # Take the two longest segments as potential MRZ lines
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
        
        # If we couldn't get proper MRZ lines, try to extract data from full text
        if not mrz_lines:
            if verbose:
                print(f"  → No valid MRZ lines found, attempting data extraction from full text")
            
            # Extract passport data from full text
            passport_data = extract_passport_data_from_text(all_text, verbose)
            
            if not passport_data.get("passport_number") and not passport_data.get("surname"):
                if verbose:
                    print(f"  ✗ Could not extract meaningful passport data")
                return {
                    "success": False,
                    "passport_data": {},
                    "mrz_text": "",
                    "method_used": "Tesseract",
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
                    "method_used": "Tesseract",
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