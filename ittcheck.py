"""
ITTCheck Passport Validation - STEP 6
Processes validation failure JSON files from temp/ folder and creates ITTMRZ output
"""

import json
import re
from PIL import Image
from typing import Dict
from datetime import datetime
from collections import Counter
from pathlib import Path

# Engine priority for tie-breaking
ENGINE_PRIORITY = [
    "FastMRZ",
    "PassportEye", 
    "EasyOCR",
    "Tesseract"
]

FIELDS = [
    "document_type",
    "country_code", 
    "surname",
    "given_names",
    "passport_number",
    "country_name",
    "country_iso",
    "nationality",
    "date_of_birth",
    "sex",
    "expiry_date",
    "personal_number"
]

ALLOWED_SEX_VALUES = {"F", "M", "0", "1", "2", "X"}

def clean_name(value: str) -> str:
    """Clean name fields by removing invalid characters"""
    if not value:
        return ""
    value = re.sub(r"[^A-Za-z\s]", "", value)
    return re.sub(r"\s+", " ", value).strip().upper()

def normalize_sex(value: str) -> str:
    """Normalize sex field to allowed values"""
    value = value.strip().upper()
    return value if value in ALLOWED_SEX_VALUES else ""

def normalize_value(field: str, value: str) -> str:
    """Normalize field values based on field type"""
    if not value:
        return ""
    
    value = value.strip()
    
    if field in ("surname", "given_names"):
        value = clean_name(value)
    elif field == "country_code":
        value = value.upper()
    elif field == "sex":
        value = normalize_sex(value)
    
    return value

def collect_valid_values(data: dict, field: str):
    """Collect valid values for a field from all engines"""
    values = []
    
    for block in data.values():
        original = block.get("original_data", {})
        errors = block.get("field_errors", {})
        
        if field in errors:
            continue
            
        if field in original:
            val = normalize_value(field, original.get(field))
            if val:
                values.append(val)
    
    return values

def resolve_given_names(values):
    """Resolve given_names using substring rule"""
    if not values:
        return ""
    
    values = list(set(values))  # remove duplicates
    
    # Substring rule
    for v in values:
        for other in values:
            if v != other and v in other and len(v) < len(other):
                return v
    
    # Fallback: least repetition and shortest
    def repetition_score(s):
        return sum(count - 1 for count in Counter(s).values())
    
    values.sort(key=lambda x: (repetition_score(x), len(x)))
    return values[0]

def majority_vote(values, data, field):
    """Perform majority vote with engine priority tie-breaking"""
    if not values:
        return ""
    
    if field == "given_names":
        return resolve_given_names(values)
    
    counter = Counter(values)
    common = counter.most_common()
    
    if len(common) == 1 or common[0][1] > common[1][1]:
        return common[0][0]
    
    # Tie-break via engine priority
    for engine in ENGINE_PRIORITY:
        block = data.get(engine, {})
        original = block.get("original_data", {})
        errors = block.get("field_errors", {})
        
        if field in original and field not in errors:
            val = normalize_value(field, original.get(field))
            if val in counter:
                return val
    
    return common[0][0]

def check_digit(value: str) -> str:
    """Compute MRZ check digit according to ICAO 9303"""
    weights = [7, 3, 1]
    char_values = {str(i): i for i in range(10)}
    char_values.update({chr(i): i-55 for i in range(65, 91)})  # A=10,...,Z=35
    char_values['<'] = 0
    
    total = 0
    for i, c in enumerate(value):
        total += char_values.get(c, 0) * weights[i % 3]
    return str(total % 10)

def format_name(surname, given_names):
    """Convert to MRZ format: SURNAME<<GIVEN<NAMES"""
    surname_mrz = re.sub(r"[^A-Z<]", "<", surname.upper())
    given_mrz = re.sub(r"[^A-Z<]", "<", given_names.upper())
    given_mrz = given_mrz.replace(" ", "<")
    return f"{surname_mrz}<<{given_mrz}"

def generate_mrz(data: dict):
    """Generate MRZ text from passport data"""
    doc_type = data.get("document_type", "P")
    country = data.get("country_code", "XXX")
    surname = data.get("surname", "")
    given_names = data.get("given_names", "")
    passport_number = data.get("passport_number", "")
    nationality = data.get("country_code", "")
    birth_date = data.get("date_of_birth", "")
    sex = data.get("sex", "<")
    expiry_date = data.get("expiry_date", "")
    personal_number = data.get("personal_number", "")
    
    # Format dates YYMMDD
    birth_mrz = birth_date[2:4] + birth_date[5:7] + birth_date[8:10] if birth_date else "000000"
    expiry_mrz = expiry_date[2:4] + expiry_date[5:7] + expiry_date[8:10] if expiry_date else "000000"
    
    # MRZ line1
    name_field = format_name(surname, given_names)
    line1 = f"{doc_type}<" + country + name_field
    line1 = line1[:44].ljust(44, '<')
    
    # MRZ line2
    passport_number_field = passport_number[:9].ljust(9, '<')
    passport_cd = check_digit(passport_number_field)
    birth_cd = check_digit(birth_mrz)
    expiry_cd = check_digit(expiry_mrz)
    personal_number_field = (personal_number[:14] if personal_number else "").ljust(14, '<')
    personal_cd = check_digit(personal_number_field)
    
    line2 = f"{passport_number_field}{passport_cd}{nationality[:3].ljust(3,'<')}{birth_mrz}{birth_cd}{sex}{expiry_mrz}{expiry_cd}{personal_number_field}{personal_cd}"
    line2 = line2[:44].ljust(44, '<')
    
    return f"{line1}\n{line2}"

def build_ittmrz(data: dict):
    """Build ITTMRZ output from validation failure data"""
    original_data = {}
    for field in FIELDS:
        values = collect_valid_values(data, field)
        original_data[field] = majority_vote(values, data, field)
    
    # Populate country_name, country_iso, and nationality based on country_code
    country_code = original_data.get("country_code", "")
    if country_code:
        try:
            from country_code import get_country_info
            country_info = get_country_info(country_code)
            
            if "error" not in country_info:
                # Update fields based on country_code
                original_data["country_name"] = country_info["name"]
                original_data["country_iso"] = country_info["alpha2"]
                original_data["nationality"] = country_info["nationality"]
            else:
                # Keep existing values if country lookup fails
                if not original_data.get("country_name"):
                    original_data["country_name"] = country_code
                if not original_data.get("country_iso"):
                    original_data["country_iso"] = country_code[:2] if len(country_code) >= 2 else ""
                if not original_data.get("nationality"):
                    original_data["nationality"] = ""
        except ImportError:
            # Fallback if country_code module is not available
            if not original_data.get("country_name"):
                original_data["country_name"] = country_code
            if not original_data.get("country_iso"):
                original_data["country_iso"] = country_code[:2] if len(country_code) >= 2 else ""
            if not original_data.get("nationality"):
                original_data["nationality"] = ""
    
    mrz_text = generate_mrz(original_data)
    
    return {
        "ITTMRZ": {
            "original_data": original_data,
            "mrz_text": mrz_text,
            "timestamp": datetime.now().isoformat()
        }
    }

def get_user_validation_file(user_id: str) -> str:
    """Get the validation failure JSON file for the current user"""
    temp_dir = Path("temp")
    
    # Look for validation failure files for this user
    validation_files = list(temp_dir.glob(f"validation_failures_{user_id}.json"))
    
    if validation_files:
        return str(validation_files[0])
    
    # If no exact match, look for any validation failure files
    all_validation_files = list(temp_dir.glob("validation_failures_*.json"))
    
    if all_validation_files:
        # Return the most recent one
        latest_file = max(all_validation_files, key=lambda f: f.stat().st_mtime)
        return str(latest_file)
    
    return ""

def validate_passport_with_ittcheck(image: Image.Image, verbose: bool = True, user_id: str = None) -> Dict:
    """
    ITTCheck Passport Validation - STEP 6
    
    Args:
        image: PIL Image object (not used in this implementation)
        verbose: Print detailed logs
        user_id: User identifier for tracking
        
    Returns:
        Dictionary with success status and passport data
    """
    try:
        if verbose:
            print(f"🔍 STEP 6: ITTCheck Processing...")
        
        if not user_id:
            if verbose:
                print(f"  ✗ ITTCheck: No user_id provided")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "method_used": "ITTCheck",
                "error": "No user_id provided for ITTCheck processing"
            }
        
        # Get validation failure JSON file for current user
        validation_file = get_user_validation_file(user_id)
        
        if not validation_file:
            if verbose:
                print(f"  ✗ ITTCheck: No validation failure file found for user {user_id}")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "method_used": "ITTCheck", 
                "error": f"No validation failure file found for user {user_id}"
            }
        
        if verbose:
            print(f"  → Loading validation data from: {Path(validation_file).name}")
        
        # Load and process validation failure data
        with open(validation_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        if verbose:
            print(f"  → Found data from engines: {list(input_data.keys())}")
        
        # Build ITTMRZ using the same logic as test/ittmrz.py
        result = build_ittmrz(input_data)
        
        # Extract data for validation
        ittmrz_data = result["ITTMRZ"]
        passport_data = ittmrz_data["original_data"]
        mrz_text = ittmrz_data["mrz_text"]
        
        if verbose:
            print(f"  → Generated ITTMRZ data:")
            print(f"    • Document: {passport_data.get('document_type', 'N/A')}")
            print(f"    • Country: {passport_data.get('country_code', 'N/A')}")
            print(f"    • Name: {passport_data.get('surname', 'N/A')}, {passport_data.get('given_names', 'N/A')}")
            print(f"    • Passport: {passport_data.get('passport_number', 'N/A')}")
            print(f"    • Sex: {passport_data.get('sex', 'N/A')}")
        
        # Perform field validation check using extracted passport data
        from utils import check_passport_data_validation_threshold
        validation_check = check_passport_data_validation_threshold(passport_data, threshold=10, verbose=verbose)
        
        if validation_check["threshold_met"]:
            if verbose:
                print(f"  ✅ ITTCheck: Field validation passed ({validation_check['valid_count']}/10)")
            
            return {
                "success": True,
                "passport_data": passport_data,
                "mrz_text": mrz_text,
                "method_used": "ITTCheck",
                "error": "",
                "ittmrz_result": result,
                "validation_summary": validation_check
            }
        else:
            if verbose:
                print(f"  ⚠️  ITTCheck: Field validation threshold not met ({validation_check['valid_count']}/10)")
            
            # Save validation failure for next step
            if user_id:
                from utils import save_validation_failure
                save_validation_failure(user_id, "ITTCheck", passport_data, validation_check["field_results"], mrz_text, "")
            
            return {
                "success": False,
                "passport_data": passport_data,
                "mrz_text": mrz_text,
                "method_used": "ITTCheck",
                "error": f"Field validation threshold not met: {validation_check['valid_count']}/10 fields valid",
                "ittmrz_result": result,
                "validation_summary": validation_check
            }
    
    except Exception as e:
        if verbose:
            print(f"  ✗ ITTCheck error: {e}")
        
        return {
            "success": False,
            "passport_data": {},
            "mrz_text": "",
            "method_used": "ITTCheck",
            "error": f"ITTCheck processing error: {str(e)}"
        }

# Alias for backward compatibility
def ittcheck_with_validation_threshold(image: Image.Image, verbose: bool = True, user_id: str = None) -> Dict:
    """
    ITTCheck with validation threshold checking (alias for main function)
    """
    return validate_passport_with_ittcheck(image, verbose, user_id)

if __name__ == "__main__":
    print("🔍 ITTCheck Passport Validation - STEP 6")
    print("=" * 50)
    
    # Test with a sample validation file
    import sys
    
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        # Find any validation failure file for testing
        temp_dir = Path("temp")
        validation_files = list(temp_dir.glob("validation_failures_*.json"))
        
        if validation_files:
            # Extract user_id from filename
            latest_file = max(validation_files, key=lambda f: f.stat().st_mtime)
            filename = latest_file.name
            # Extract user_id from filename like "validation_failures_base64_20251217_185423_344314_9aadf93e.json"
            user_id = filename.replace("validation_failures_", "").replace(".json", "")
            print(f"Using user_id from latest file: {user_id}")
        else:
            print("No validation failure files found in temp/ folder")
            print("Usage: python ittcheck.py [user_id]")
            sys.exit(1)
    
    # Test the ITTCheck function
    result = validate_passport_with_ittcheck(None, verbose=True, user_id=user_id)
    
    print("\n" + "=" * 50)
    print("RESULT:")
    print(f"Success: {result.get('success', False)}")
    
    if result.get("success"):
        print("✅ ITTCheck processing completed successfully!")
        ittmrz_result = result.get("ittmrz_result", {})
        if ittmrz_result:
            print("\nITTMRZ Output:")
            print(json.dumps(ittmrz_result, indent=2, ensure_ascii=False))
    else:
        print(f"❌ ITTCheck failed: {result.get('error', 'Unknown error')}")
        
        # Still show the ITTMRZ result if available
        ittmrz_result = result.get("ittmrz_result", {})
        if ittmrz_result:
            print("\nGenerated ITTMRZ (validation failed):")
            print(json.dumps(ittmrz_result, indent=2, ensure_ascii=False))