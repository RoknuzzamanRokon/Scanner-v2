"""
PassportEye fallback validation with comprehensive TD3 support
"""
import cv2
import numpy as np
from PIL import Image
from typing import Dict
from passporteye import read_mrz
from country_code import get_country_info
from utils import format_mrz_date


def validate_passport_with_PassportEye_fallback(image: Image.Image, verbose: bool = True) -> Dict:
    """
    STEP 1: PassportEye Fallback Validation
    
    - Image preprocessing for better OCR
    - PassportEye MRZ detection
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
            print(f"  → Processing with PassportEye...")
        
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocessing for better OCR
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Upscale for better OCR
        gray = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2))
        
        # Apply adaptive thresholding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Save preprocessed image temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, thresh)
            temp_path = tmp.name
        
        try:
            # PassportEye MRZ detection
            if verbose:
                print(f"  → Running PassportEye MRZ detection...")
            
            mrz = read_mrz(temp_path)
            
            if not mrz:
                if verbose:
                    print(f"  ✗ PassportEye: No MRZ detected")
                return {
                    "success": False,
                    "passport_data": {},
                    "mrz_text": "",
                    "method_used": "PassportEye",
                    "error": "No MRZ detected by PassportEye"
                }
            
            # Extract MRZ data
            mrz_data = mrz.to_dict()
            
            if verbose:
                print(f"  ✓ MRZ detected")
                print(f"    Type: {mrz_data.get('mrz_type', 'Unknown')}")
                print(f"    Valid Score: {mrz_data.get('valid_score', 0)}")
            
            # Get raw MRZ text
            raw_text = mrz_data.get('raw_text', '')
            
            if not raw_text:
                if verbose:
                    print(f"  ✗ No MRZ text extracted")
                return {
                    "success": False,
                    "passport_data": {},
                    "mrz_text": "",
                    "method_used": "PassportEye",
                    "error": "No MRZ text extracted"
                }
            
            # Clean and validate TD3 format
            mrz_lines = raw_text.strip().split('\n')
            
            if len(mrz_lines) != 2:
                if verbose:
                    print(f"  ✗ Invalid MRZ format: Expected 2 lines, got {len(mrz_lines)}")
                return {
                    "success": False,
                    "passport_data": {},
                    "mrz_text": raw_text,
                    "method_used": "PassportEye",
                    "error": f"Invalid MRZ format: Expected 2 lines, got {len(mrz_lines)}"
                }
            
            # TD3 Cleaning Rules
            line1 = mrz_lines[0]
            line2 = mrz_lines[1]
            
            # Fix Line 1: Ensure exactly P< (not P<<)
            if line1.startswith('P<<'):
                line1 = 'P<' + line1[3:]
            
            # Ensure both lines are exactly 44 characters
            line1 = line1[:44].ljust(44, '<')
            line2 = line2[:44].ljust(44, '<')
            
            cleaned_mrz_text = f"{line1}\n{line2}"
            
            
            if verbose:
                print(f"  ✓ MRZ cleaned and validated")
                print(f"    Line 1: {line1}")
                print(f"    Line 2: {line2}")
            
            # Validate MRZ using TD3 validation checker
            from passport_detector import passport_validation_checker
            
            if verbose:
                print(f"  → Validating MRZ with TD3 rules...")
            
            validation_result = passport_validation_checker(cleaned_mrz_text, verbose=False)
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
                    "mrz_text": cleaned_mrz_text,
                    "method_used": "PassportEye",
                    "error": f"MRZ validation failed: {validation_result.get('reason', 'Low confidence')} (confidence: {confidence*100:.1f}%)"
                }

            
            # Extract passport fields
            try:
                document_type = mrz_data.get('type', 'P')
                country_code = mrz_data.get('country', '').upper()
                surname = mrz_data.get('surname', '').upper()
                names = mrz_data.get('names', '').upper()
                passport_number = mrz_data.get('number', '')
                nationality = mrz_data.get('nationality', '').upper()
                date_of_birth = mrz_data.get('date_of_birth', '')
                sex = mrz_data.get('sex', '')
                expiration_date = mrz_data.get('expiration_date', '')
                personal_number = mrz_data.get('personal_number', '')
                
                # Clean country code (remove symbols)
                country_code_clean = ''.join([c for c in country_code if c.isalpha()])[:3]
                nationality_clean = ''.join([c for c in nationality if c.isalpha()])[:3]
                
                # Get country details
                country_info = get_country_info(country_code_clean)
                country_name = country_info.get('name', country_code_clean)
                country_iso = country_info.get('alpha2', '')
                nationality_name = country_info.get('nationality', nationality_clean)
                
                # Format dates
                dob_formatted = format_mrz_date(date_of_birth) if date_of_birth else ""
                expiry_formatted = format_mrz_date(expiration_date) if expiration_date else ""
                
                # Validate we have meaningful data
                has_meaningful_data = any([
                    surname,
                    names,
                    country_code_clean,
                    date_of_birth,
                    expiration_date
                ])
                
                if not has_meaningful_data:
                    if verbose:
                        print(f"  ✗ No meaningful data extracted")
                    return {
                        "success": False,
                        "passport_data": {},
                        "mrz_text": cleaned_mrz_text,
                        "method_used": "PassportEye",
                        "error": "No meaningful passport data extracted"
                    }
                
                # Build passport data
                passport_data = {
                    "document_type": document_type,
                    "country_code": country_code_clean,
                    "surname": surname,
                    "given_names": names,
                    "passport_number": passport_number,
                    "country_name": country_name,
                    "country_iso": country_iso,
                    "nationality": nationality_name,
                    "date_of_birth": dob_formatted,
                    "sex": sex if sex in ['M', 'F', 'X', '<'] else '<',
                    "expiry_date": expiry_formatted,
                    "personal_number": personal_number.replace('<', '').strip()
                }
                
                if verbose:
                    print(f"  ✓ Passport data extracted successfully")
                    print(f"    Surname: {surname}")
                    print(f"    Given Names: {names}")
                    print(f"    Passport #: {passport_number}")
                    print(f"    Country: {country_name} ({country_code_clean})")
                
                return {
                    "success": True,
                    "passport_data": passport_data,
                    "mrz_text": cleaned_mrz_text,
                    "method_used": "PassportEye",
                    "error": ""
                }
                
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error parsing MRZ data: {e}")
                return {
                    "success": False,
                    "passport_data": {},
                    "mrz_text": cleaned_mrz_text,
                    "method_used": "PassportEye",
                    "error": f"Error parsing MRZ data: {str(e)}"
                }
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        import traceback
        if verbose:
            print(f"  ✗ PassportEye error: {e}")
            traceback.print_exc()
        
        return {
            "success": False,
            "passport_data": {},
            "mrz_text": "",
            "method_used": "PassportEye",
            "error": f"PassportEye processing error: {str(e)}"
        }
