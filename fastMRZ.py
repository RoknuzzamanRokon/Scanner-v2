"""
FastMRZ fallback validation with comprehensive error handling
"""
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from typing import Dict
from fastmrz import FastMRZ
from country_code import get_country_info
from utils import format_mrz_date


def validate_passport_with_fastmrz_fallback(image: Image.Image, verbose: bool = True) -> Dict:
    """
    STEP 2: FastMRZ Fallback Validation
    
    - FastMRZ library integration
    - Format output to standard response structure
    - Comprehensive error handling
    
    Args:
        image: PIL Image object
        verbose: Print detailed logs
        
    Returns:
        Dictionary with success status and passport data
    """
    try:
        if verbose:
            print(f"  → Processing with FastMRZ...")
        
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Save to temporary file (FastMRZ requires file path)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img_cv)
            temp_path = tmp.name
        
        try:
            # Initialize FastMRZ
            fast_mrz = FastMRZ()
            
            if verbose:
                print(f"  → Running FastMRZ detection...")
            
            # Extract MRZ details
            details = fast_mrz.get_details(temp_path, include_checkdigit=False)
            
            if not details:
                if verbose:
                    print(f"  ✗ FastMRZ: No MRZ detected")
                return {
                    "success": False,
                    "passport_data": {},
                    "mrz_text": "",
                    "method_used": "FastMRZ",
                    "error": "No MRZ detected by FastMRZ"
                }
            
            if verbose:
                print(f"  ✓ MRZ detected by FastMRZ")
                print(f"    Type: {details.get('mrz_type', 'Unknown')}")
            
            # Check for required fields
            if not details.get('document_number'):
                if verbose:
                    print(f"  ✗ No document number found")
                return {
                    "success": False,
                    "passport_data": {},
                    "mrz_text": "",
                    "method_used": "FastMRZ",
                    "error": "No document number extracted"
                }
            
            # Extract fields
            document_type = details.get('document_code', 'P')
            country_code = details.get('issuer_code', '').upper()
            surname = details.get('surname', '').replace('<', ' ').strip().upper()
            given_names = details.get('given_name', '').replace('<', ' ').strip().upper()
            passport_number = details.get('document_number', '')
            nationality_code = details.get('nationality_code', '').upper()
            date_of_birth = details.get('birth_date', '')  # Already in YYYY-MM-DD format
            sex = details.get('sex', '<').upper()
            expiry_date = details.get('expiry_date', '')  # Already in YYYY-MM-DD format
            personal_number = details.get('optional_data', '').replace('<', '').strip()
            
            # Clean country codes (remove symbols)
            country_code_clean = ''.join([c for c in country_code if c.isalpha()])[:3]
            nationality_clean = ''.join([c for c in nationality_code if c.isalpha()])[:3]
            
            # Get country details
            country_info = get_country_info(country_code_clean)
            country_name = country_info.get('name', country_code_clean)
            country_iso = country_info.get('alpha2', '')
            nationality_name = country_info.get('nationality', nationality_clean)
            
            # Validate we have meaningful data
            has_meaningful_data = any([
                surname,
                given_names,
                country_code_clean,
                date_of_birth,
                expiry_date
            ])
            
            if not has_meaningful_data:
                if verbose:
                    print(f"  ✗ No meaningful data extracted")
                return {
                    "success": False,
                    "passport_data": {},
                    "mrz_text": "",
                    "method_used": "FastMRZ",
                    "error": "No meaningful passport data extracted"
                }
            
            # Build passport data
            passport_data = {
                "document_type": document_type,
                "country_code": country_code_clean,
                "surname": surname,
                "given_names": given_names,
                "passport_number": passport_number,
                "country_name": country_name,
                "country_iso": country_iso,
                "nationality": nationality_name,
                "date_of_birth": date_of_birth,
                "sex": sex if sex in ['M', 'F', 'X', '<'] else '<',
                "expiry_date": expiry_date,
                "personal_number": personal_number
            }
            
            # Construct MRZ text
            mrz_text = details.get('raw_text', '')
            if not mrz_text:
                # Reconstruct MRZ from parsed data if raw_text not available
                line1 = f"P<{country_code_clean}{surname}<<{given_names.replace(' ', '<')}"
                line1 = line1[:44].ljust(44, '<')
                
                line2 = f"{passport_number.ljust(9, '<')}<{nationality_clean}{date_of_birth.replace('-', '')[2:]}<{sex}{expiry_date.replace('-', '')[2:]}<{personal_number.ljust(14, '<')}<"
                line2 = line2[:44].ljust(44, '<')
                
                
                mrz_text = f"{line1}\n{line2}"
            
            # Validate MRZ using TD3 validation checker
            from passport_detector import passport_validation_checker
            
            if verbose:
                print(f"  → Validating MRZ with TD3 rules...")
            
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
                    "method_used": "FastMRZ",
                    "error": f"MRZ validation failed: {validation_result.get('reason', 'Low confidence')} (confidence: {confidence*100:.1f}%)"
                }

            
            if verbose:
                print(f"  ✓ Passport data extracted successfully")
                print(f"    Surname: {surname}")
                print(f"    Given Names: {given_names}")
                print(f"    Passport #: {passport_number}")
                print(f"    Country: {country_name} ({country_code_clean})")
            
            return {
                "success": True,
                "passport_data": passport_data,
                "mrz_text": mrz_text,
                "method_used": "FastMRZ",
                "error": ""
            }
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        import traceback
        if verbose:
            print(f"  ✗ FastMRZ error: {e}")
            traceback.print_exc()
        
        return {
            "success": False,
            "passport_data": {},
            "mrz_text": "",
            "method_used": "FastMRZ",
            "error": f"FastMRZ processing error: {str(e)}"
        }
