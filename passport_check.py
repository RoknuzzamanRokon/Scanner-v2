"""
Individual field validation for TD3 MRZ passport documents
Returns validation status for each field separately
"""
import re
from datetime import datetime
from typing import Dict, Any


def validate_passport_fields(mrz_text: str) -> Dict[str, str]:
    """
    Validate each field in TD3 MRZ format individually
    
    Args:
        mrz_text: Two-line MRZ text (44 characters each line)
        
    Returns:
        Dictionary with validation status for each field:
        - "Valid": Field passes validation
        - "Invalid": Field fails validation
        - "Missing": Field is missing or empty
        
    Example:
        {
            "document_type": "Valid",
            "issuing_country": "Valid", 
            "surname": "Valid",
            "given_names": "Valid",
            "passport_number": "Valid",
            "nationality": "Valid",
            "date_of_birth": "Invalid",
            "sex": "Valid",
            "expiry_date": "Valid",
            "personal_number": "Valid"
        }
    """
    
    # Initialize result with all fields as Invalid
    result = {
        "document_type": "Invalid",
        "issuing_country": "Invalid",
        "surname": "Invalid", 
        "given_names": "Invalid",
        "passport_number": "Invalid",
        "nationality": "Invalid",
        "date_of_birth": "Invalid",
        "sex": "Invalid",
        "expiry_date": "Invalid",
        "personal_number": "Invalid"
    }
    
    try:
        # Split MRZ into two lines
        lines = mrz_text.strip().split('\n')
        if len(lines) != 2:
            return result  # All fields remain Invalid
        
        line1 = lines[0].strip()
        line2 = lines[1].strip()
        
        # Check line lengths
        if len(line1) != 44 or len(line2) != 44:
            return result  # All fields remain Invalid
        
        # Validate Line 1 fields
        result.update(_validate_line1_fields(line1))
        
        # Validate Line 2 fields  
        result.update(_validate_line2_fields(line2))
        
        return result
        
    except Exception:
        return result  # All fields remain Invalid


def _validate_line1_fields(line1: str) -> Dict[str, str]:
    """Validate fields in Line 1 of TD3 MRZ"""
    result = {}
    
    # Document Type (Position 0-1)
    doc_type = line1[0:1]
    if re.match(r'^[A-Z]$', doc_type) and doc_type in ['P', 'PO', 'PD', 'PN', 'PS']:
        result["document_type"] = "Valid"
    else:
        result["document_type"] = "Invalid"
    
    # Issuing Country (Position 2-4)
    country = line1[2:5]
    if re.match(r'^[A-Z]{3}$', country) and country != '<<<':
        result["issuing_country"] = "Valid"
    else:
        result["issuing_country"] = "Invalid"
    
    # Name field (Position 5-43) - Contains surname and given names
    name_field = line1[5:44]
    surname_result, given_names_result = _validate_name_field(name_field)
    result["surname"] = surname_result
    result["given_names"] = given_names_result
    
    return result


def _validate_line2_fields(line2: str) -> Dict[str, str]:
    """Validate fields in Line 2 of TD3 MRZ"""
    result = {}
    
    # Passport Number (Position 0-8)
    passport_num = line2[0:9].rstrip('<')
    if len(passport_num) >= 1 and re.match(r'^[A-Z0-9]+$', passport_num):
        result["passport_number"] = "Valid"
    else:
        result["passport_number"] = "Invalid"
    
    # Nationality (Position 10-12)
    nationality = line2[10:13]
    if re.match(r'^[A-Z]{3}$', nationality) and nationality != '<<<':
        result["nationality"] = "Valid"
    else:
        result["nationality"] = "Invalid"
    
    # Date of Birth (Position 13-18)
    birth_date = line2[13:19]
    result["date_of_birth"] = _validate_date_field(birth_date, "birth")
    
    # Sex (Position 20)
    sex = line2[20:21]
    
    # Normalize sex field (convert 0->M, 1->F, etc.)
    from sex_field_normalizer import normalize_sex_field
    normalized_sex = normalize_sex_field(sex)
    
    if normalized_sex in ['M', 'F', 'X', '<']:
        result["sex"] = "Valid"
    else:
        result["sex"] = "Invalid"
    
    # Expiry Date (Position 21-26)
    expiry_date = line2[21:27]
    result["expiry_date"] = _validate_date_field(expiry_date, "expiry")
    
    # Personal Number (Position 28-41) - Optional field
    personal_num = line2[28:42].rstrip('<')
    if len(personal_num) == 0:
        # Empty personal number is valid (optional field)
        result["personal_number"] = "Valid"
    elif re.match(r'^[A-Z0-9]+$', personal_num):
        result["personal_number"] = "Valid"
    else:
        result["personal_number"] = "Invalid"
    
    return result


def _validate_name_field(name_field: str) -> tuple:
    """
    Validate the name field which contains surname and given names
    Format: SURNAME<<GIVEN<NAMES<<<<<<<<<<
    
    Returns:
        Tuple of (surname_status, given_names_status)
    """
    try:
        # Split by double chevrons to separate surname from given names
        if '<<' not in name_field:
            return ("Invalid", "Invalid")
        
        parts = name_field.split('<<', 1)
        surname_part = parts[0]
        given_names_part = parts[1] if len(parts) > 1 else ""
        
        # Validate surname
        surname_status = "Invalid"
        if len(surname_part) >= 1 and re.match(r'^[A-Z]+$', surname_part):
            surname_status = "Valid"
        
        # Validate given names
        given_names_status = "Invalid"
        if len(given_names_part) >= 1:
            # Remove trailing chevrons and split by single chevrons
            given_names_clean = given_names_part.rstrip('<')
            if len(given_names_clean) >= 1:
                # Check that it contains only valid characters (letters, chevrons, and numbers)
                # Some MRZ formats may include numbers in the name field
                if re.match(r'^[A-Z0-9<]+$', given_names_clean):
                    # Check that it contains at least one letter
                    if re.search(r'[A-Z]', given_names_clean):
                        given_names_status = "Valid"
        elif len(given_names_part.rstrip('<')) == 0:
            # Empty given names part (only chevrons) - some passports have only surname
            given_names_status = "Valid"
        
        return (surname_status, given_names_status)
        
    except Exception:
        return ("Invalid", "Invalid")


def _validate_date_field(date_str: str, date_type: str) -> str:
    """
    Validate date field in YYMMDD format
    
    Args:
        date_str: 6-character date string
        date_type: "birth" or "expiry" for context
        
    Returns:
        "Valid" or "Invalid"
    """
    try:
        if len(date_str) != 6:
            return "Invalid"
        
        if not re.match(r'^\d{6}$', date_str):
            return "Invalid"
        
        year = int(date_str[0:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        
        # Basic range checks
        if month < 1 or month > 12:
            return "Invalid"
        
        if day < 1 or day > 31:
            return "Invalid"
        
        # Days per month validation
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Allowing Feb 29 for simplicity
        if day > days_in_month[month - 1]:
            return "Invalid"
        
        # Convert 2-digit year to 4-digit year
        current_year = datetime.now().year
        current_year_2digit = current_year % 100
        
        if date_type == "birth":
            # Birth dates: more intelligent year conversion
            # If year is greater than current year + 5, assume 1900s
            # This handles cases where someone born in 1925 has year "25" in their passport
            if year > current_year_2digit + 5:
                full_year = 1900 + year
            else:
                # For years <= current_year + 5, check if it would be a future date
                test_2000s = 2000 + year
                if test_2000s > current_year:
                    # Future date in 2000s, so it must be 1900s
                    full_year = 1900 + year
                else:
                    # Valid date in 2000s
                    full_year = 2000 + year
        else:  # expiry
            # Expiry dates: if year < current_year_2digit - 10, assume 2100s, else 2000s
            if year < current_year_2digit - 10:
                full_year = 2100 + year
            else:
                full_year = 2000 + year
        
        # Create datetime object to validate the date
        test_date = datetime(full_year, month, day)
        
        # Additional logical checks
        if date_type == "birth":
            current_date = datetime.now()
            
            # Birth date should not be in the future
            if test_date > current_date:
                return "Invalid"
            
            # Birth date must be at least 1 month older than current date
            # Calculate date 1 month ago from today
            current_year = current_date.year
            current_month = current_date.month
            current_day = current_date.day
            
            # Calculate 1 month ago
            if current_month == 1:
                one_month_ago_year = current_year - 1
                one_month_ago_month = 12
            else:
                one_month_ago_year = current_year
                one_month_ago_month = current_month - 1
            
            # Handle day overflow (e.g., if today is March 31, one month ago would be Feb 28/29)
            try:
                one_month_ago = datetime(one_month_ago_year, one_month_ago_month, current_day)
            except ValueError:
                # Day doesn't exist in previous month (e.g., March 31 -> Feb 31 doesn't exist)
                # Use the last day of the previous month
                import calendar
                last_day = calendar.monthrange(one_month_ago_year, one_month_ago_month)[1]
                one_month_ago = datetime(one_month_ago_year, one_month_ago_month, last_day)
            
            if test_date > one_month_ago:
                return "Invalid"
            
            # Birth date should not be more than 150 years ago
            if test_date.year < current_date.year - 150:
                return "Invalid"
        
        elif date_type == "expiry":
            # Expiry date should not be more than 50 years in the future
            if test_date.year > datetime.now().year + 50:
                return "Invalid"
            # Expiry date should not be more than 20 years in the past
            if test_date.year < datetime.now().year - 20:
                return "Invalid"
        
        return "Valid"
        
    except (ValueError, TypeError):
        return "Invalid"


def _validate_checksum(data: str, check_digit: str) -> bool:
    """
    Validate MRZ checksum digit
    
    Args:
        data: Data string to validate
        check_digit: Expected check digit
        
    Returns:
        True if checksum is valid
    """
    try:
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
                return False
            
            total += value * weights[i % 3]
        
        calculated_check = str(total % 10)
        return calculated_check == check_digit
        
    except Exception:
        return False


# # Example usage and testing
# if __name__ == "__main__":
#     print("PASSPORT FIELD VALIDATION TESTER")
#     print("=" * 50)
    
#     # Test with Syrian passport MRZ
#     print("\n1. Testing Syrian Passport MRZ:")
#     print("-" * 30)
#     sample_mrz1 = """P<UTOQUEST<<AMELIA<MARY<<<<<<<<<<<<<<<<<<<<<
# S012345674UTO8704234F3208265<<<<<<<<<<<<<<<2"""
    
#     result1 = validate_passport_fields(sample_mrz1)
    
#     for field, status in result1.items():
#         status_icon = "✅" if status == "Valid" else "❌"
#         print(f"{status_icon} {field:20}: {status}")
    
#     # Test with Bangladesh passport MRZ
#     print("\n2. Testing Bangladesh Passport MRZ:")
#     print("-" * 30)
#     sample_mrz2 = """P<UTOQUEST<<AMELIA<MARY<<<<<<<<<<<<<<<<<<<<<
# S012345674UTO8704234F3208265<<<<<<<<<<<<<<<2"""
    
#     result2 = validate_passport_fields(sample_mrz2)
    
#     for field, status in result2.items():
#         status_icon = "✅" if status == "Valid" else "❌"
#         print(f"{status_icon} {field:20}: {status}")
    
#     # Test with invalid MRZ
#     print("\n3. Testing Invalid MRZ (wrong format):")
#     print("-" * 30)
#     invalid_mrz = """INVALID_MRZ_FORMAT
# ALSO_INVALID_FORMAT"""
    
#     result3 = validate_passport_fields(invalid_mrz)
    
#     for field, status in result3.items():
#         status_icon = "✅" if status == "Valid" else "❌"
#         print(f"{status_icon} {field:20}: {status}")
    
#     print("\n" + "=" * 50)
#     print("Testing completed!")