"""
Passport validation checker with TD3 compliance and confidence scoring
"""
import re
from typing import Dict
from td3_validation_check import TD3_MRZ_RULES


def passport_validation_checker(mrz_text: str, verbose: bool = True) -> Dict:
    """
    STEP 3: Passport Validation Checker
    
    - MRZ text validation against TD3 standards
    - Confidence scoring (0.0 to 1.0)
    - Threshold-based validation (≥50% confidence)
    
    Args:
        mrz_text: MRZ text to validate (2 lines, 44 chars each)
        verbose: Print detailed logs
        
    Returns:
        Dictionary with validation results and confidence score
    """
    try:
        if verbose:
            print(f"  → Validating MRZ against TD3 standards...")
        
        if not mrz_text or not isinstance(mrz_text, str):
            if verbose:
                print(f"  ✗ Invalid MRZ text: Empty or non-string")
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "reason": "Empty or invalid MRZ text",
                "passport_data": {}
            }
        
        # Split into lines
        mrz_lines = mrz_text.strip().split('\n')
        
        if len(mrz_lines) != 2:
            if verbose:
                print(f"  ✗ Invalid MRZ format: Expected 2 lines, got {len(mrz_lines)}")
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "reason": f"Invalid line count: Expected 2, got {len(mrz_lines)}",
                "passport_data": {}
            }
        
        line1 = mrz_lines[0]
        line2 = mrz_lines[1]
        
        # Validate line lengths
        if len(line1) != 44 or len(line2) != 44:
            if verbose:
                print(f"  ✗ Invalid line lengths: Line1={len(line1)}, Line2={len(line2)} (expected 44)")
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "reason": f"Invalid line lengths: {len(line1)}, {len(line2)} (expected 44 each)",
                "passport_data": {}
            }
        
        # TD3 Validation Rules
        validation_score = 0
        max_score = 0
        issues = []
        
        # LINE 1 Validation
        line1_rules = TD3_MRZ_RULES["line1"]["fields"]
        
        # Document Type (pos 0-1)
        max_score += 1
        doc_type = line1[0:2]
        if re.match(r'^P[<A-Z]$', doc_type):
            validation_score += 1
        else:
            issues.append(f"Invalid document type: {doc_type}")
        
        # Country Code (pos 2-4)
        max_score += 1
        country_code = line1[2:5]
        if re.match(r'^[A-Z]{3}$', country_code.replace('<', 'X')):
            validation_score += 1
        else:
            issues.append(f"Invalid country code: {country_code}")
        
        # Name field (pos 5-43) - should contain at least one name
        max_score += 1
        name_field = line1[5:44]
        if '<<' in name_field and any(c.isalpha() for c in name_field):
            validation_score += 1
        else:
            issues.append(f"Invalid name field format")
        
        # LINE 2 Validation
        line2_rules = TD3_MRZ_RULES["line2"]["fields"]
        
        # Passport Number (pos 0-8)
        max_score += 1
        passport_num = line2[0:9].replace('<', '')
        if passport_num:  # Can be empty per TD3 standards
            validation_score += 1
        
        # Nationality (pos 10-12)
        max_score += 1
        nationality = line2[10:13]
        if re.match(r'^[A-Z]{3}$', nationality.replace('<', 'X')):
            validation_score += 1
        else:
            issues.append(f"Invalid nationality: {nationality}")
        
        # Date of Birth (pos 13-18)
        max_score += 1
        dob = line2[13:19]
        if re.match(r'^\\d{6}$', dob):
            validation_score += 1
        else:
            issues.append(f"Invalid date of birth: {dob}")
        
        # Sex (pos 20)
        max_score += 1
        sex = line2[20]
        
        # Normalize sex field (handle 0->M, 1->F mapping)
        from sex_field_normalizer import normalize_sex_field
        normalized_sex = normalize_sex_field(sex)
        
        if normalized_sex in ['M', 'F', 'X', '<']:
            validation_score += 1
        else:
            issues.append(f"Invalid sex field: {sex}")
        
        # Expiry Date (pos 21-26)
        max_score += 1
        expiry = line2[21:27]
        if re.match(r'^\\d{6}$', expiry):
            validation_score += 1
        else:
            issues.append(f"Invalid expiry date: {expiry}")
        
        # Calculate confidence score
        confidence = validation_score / max_score if max_score > 0 else 0.0
        
        # Determine if valid (≥50% confidence)
        is_valid = confidence >= 0.5
        
        if verbose:
            print(f"  → Validation Score: {validation_score}/{max_score}")
            print(f"  → Confidence: {confidence*100:.1f}%")
            print(f"  → Valid: {is_valid}")
            if issues:
                print(f"  → Issues found:")
                for issue in issues:
                    print(f"    - {issue}")
        
        # Extract basic passport data for return
        passport_data = {}
        if is_valid:
            # Parse name field
            name_parts = name_field.split('<<')
            surname = name_parts[0].replace('<', ' ').strip() if len(name_parts) > 0 else ""
            given_names = name_parts[1].replace('<', ' ').strip() if len(name_parts) > 1 else ""
            
            # Normalize sex field (convert 0->M, 1->F, etc.)
            from sex_field_normalizer import normalize_sex_field
            normalized_sex = normalize_sex_field(sex)
            
            passport_data = {
                "document_type": doc_type[0] if doc_type else "P",
                "country_code": country_code,
                "surname": surname,
                "given_names": given_names,
                "passport_number": passport_num,
                "nationality": nationality,
                "date_of_birth": dob,
                "sex": normalized_sex,
                "expiry_date": expiry,
                "personal_number": line2[28:42].replace('<', '').strip()
            }
        
        return {
            "is_valid": is_valid,
            "confidence_score": confidence,
            "reason": " | ".join(issues) if issues else "MRZ validation passed",
            "passport_data": passport_data
        }
    
    except Exception as e:
        import traceback
        if verbose:
            print(f"  ✗ Validation error: {e}")
            traceback.print_exc()
        
        return {
            "is_valid": False,
            "confidence_score": 0.0,
            "reason": f"Validation error: {str(e)}",
            "passport_data": {}
        }


def is_passport_image(text: str, verbose: bool = False) -> bool:
    """
    Check if image contains passport-like content
    
    Rules:
    - Contains '<' symbol more than 5 times (MRZ indicator)
    - Contains 'Passport' or 'PASSPORT' keyword
    - Contains country codes or MRZ patterns
    
    Args:
        text: OCR extracted text from image
        verbose: Print detailed logs
        
    Returns:
        True if likely a passport image
    """
    if not text:
        return False
    
    # Rule 1: Check for MRZ indicators (< symbols)
    angle_bracket_count = text.count('<')
    if angle_bracket_count > 5:
        if verbose:
            print(f"  ✓ MRZ detected: {angle_bracket_count} '<' symbols found")
        return True
    
    # Rule 2: Check for passport keywords
    passport_keywords = ['passport', 'PASSPORT', 'Passeport', 'Passaporto', 'Pasaporte', 'Reisepass']
    for keyword in passport_keywords:
        if keyword in text:
            if verbose:
                print(f"  ✓ Passport keyword found: {keyword}")
            return True
    
    # Rule 3: Check for MRZ pattern (lines starting with P<)
    if re.search(r'P<[A-Z]{3}', text):
        if verbose:
            print(f"  ✓ MRZ pattern detected (P<XXX)")
        return True
    
    if verbose:
        print(f"  ✗ No passport indicators found")
    
    return False