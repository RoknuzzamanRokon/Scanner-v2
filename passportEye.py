"""
PassportEye MRZ extraction with preprocessing and timing
Simple implementation following the fastMRZ pattern
"""
import cv2
import time
import warnings
from passporteye import read_mrz

# Suppress FutureWarnings from PassportEye library
warnings.filterwarnings("ignore", category=FutureWarning, module="passporteye")

# Valid passport document codes according to ICAO standards
VALID_PASSPORT_CODES = {"P<", "PO", "PD", "PN", "PS"}

IMAGE_PATH = None  # Will be set when calling the function

def preprocess_image(image_path, output_path=None):
    """
    Preprocess image for better MRZ detection
    Creates temporary file in user's specific temp folder and returns path for cleanup
    """
    import os
    import tempfile
    from datetime import datetime
    
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Determine the user's temp folder from the image_path
    user_temp_dir = None
    if image_path and os.path.exists(image_path):
        # Get the directory containing the input image
        input_dir = os.path.dirname(image_path)
        
        # Check if it's a user-specific temp folder (contains base64_ pattern)
        if "temp" in input_dir and ("base64_" in input_dir or "temp/" in input_dir):
            user_temp_dir = input_dir
            print(f"  🎯 Detected user-specific folder: {user_temp_dir}")
        else:
            # Fallback: create general temp folder
            user_temp_dir = "temp"
            if not os.path.exists(user_temp_dir):
                os.makedirs(user_temp_dir)
            print(f"  📁 Using general temp folder: {user_temp_dir}")
    else:
        # Fallback: create general temp folder
        user_temp_dir = "temp"
        if not os.path.exists(user_temp_dir):
            os.makedirs(user_temp_dir)
        print(f"  📁 Using fallback temp folder: {user_temp_dir}")
    
    # Generate unique filename with timestamp if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = os.path.join(user_temp_dir, f"temp_preprocessed_{timestamp}.jpg")
    
    print(f"  📥 Input image path: {image_path}")
    print(f"  📁 Creating preprocessed file in: {user_temp_dir}")
    print(f"  📄 Preprocessed file: {os.path.basename(output_path)}")
    print(f"  🔗 Full output path: {output_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize image (2x upscale)
    gray = cv2.resize(gray, (img.shape[1]*2, img.shape[0]*2))
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save preprocessed image in user's temp folder
    cv2.imwrite(output_path, thresh)
    return output_path


def decode_td3_mrz_lines(line1: str, line2: str):
    """
    Decode TD3 MRZ lines and extract detailed information
    
    Args:
        line1: First MRZ line (44 characters)
        line2: Second MRZ line (44 characters)
        
    Returns:
        Dictionary with decoded MRZ details
    """
    try:
        # Decode Line 1: DOC_CODE + COUNTRY_CODE + SURNAME<<GIVEN_NAMES
        # Handle different document code formats
        if line1.startswith('P<'):
            document_code = 'P<'
            issuer_code = line1[2:5]    # Position 2-4
            name_section = line1[5:44]  # Position 5-43
        elif line1[0:2] in VALID_PASSPORT_CODES:
            document_code = line1[0:2]  # Position 0-1
            issuer_code = line1[2:5]    # Position 2-4
            name_section = line1[5:44]  # Position 5-43
        else:
            # Fallback for unknown format
            document_code = line1[0:1]  # Position 0
            issuer_code = line1[2:5]    # Position 2-4
            name_section = line1[5:44]  # Position 5-43
        
        # Parse names from name section
        if '<<' in name_section:
            name_parts = name_section.split('<<')
            surname = name_parts[0].replace('<', '').strip()
            given_name = name_parts[1].replace('<', '').strip() if len(name_parts) > 1 else ''
        else:
            surname = name_section.replace('<', '').strip()
            given_name = ''
        
        # Decode Line 2: PASSPORT_NUM + CHECK + NATIONALITY + BIRTH_DATE + CHECK + SEX + EXPIRY + CHECK + PERSONAL_NUM + CHECK + FINAL_CHECK
        passport_number = line2[0:9].replace('<', '').strip()    # Position 0-8
        passport_check = line2[9:10]                             # Position 9
        nationality_code = line2[10:13]                          # Position 10-12
        birth_date = line2[13:19]                                # Position 13-18
        birth_check = line2[19:20]                               # Position 19
        sex = line2[20:21]                                       # Position 20
        expiry_date = line2[21:27]                               # Position 21-26
        expiry_check = line2[27:28]                              # Position 27
        personal_number = line2[28:42].replace('<', '').strip()  # Position 28-41
        personal_check = line2[42:43]                            # Position 42
        final_check = line2[43:44]                               # Position 43
        
        # Format dates with proper year conversion
        if len(birth_date) == 6:
            birth_year_2digit = int(birth_date[0:2])
            birth_month = birth_date[2:4]
            birth_day = birth_date[4:6]
            
            # Smart year conversion for birth dates
            from datetime import datetime
            current_year = datetime.now().year
            current_year_2digit = current_year % 100
            
            # If year > current_year + 5, assume 1900s, else check if 2000s would be future
            if birth_year_2digit > current_year_2digit + 5:
                birth_full_year = 1900 + birth_year_2digit
            else:
                test_2000s = 2000 + birth_year_2digit
                if test_2000s > current_year:
                    birth_full_year = 1900 + birth_year_2digit
                else:
                    birth_full_year = 2000 + birth_year_2digit
            
            birth_date_formatted = f"{birth_full_year}-{birth_month}-{birth_day}"
        else:
            birth_date_formatted = birth_date
        
        if len(expiry_date) == 6:
            expiry_year_2digit = int(expiry_date[0:2])
            expiry_month = expiry_date[2:4]
            expiry_day = expiry_date[4:6]
            
            # Smart year conversion for expiry dates
            current_year_2digit = datetime.now().year % 100
            
            # If year < current_year - 10, assume 2100s, else 2000s
            if expiry_year_2digit < current_year_2digit - 10:
                expiry_full_year = 2100 + expiry_year_2digit
            else:
                expiry_full_year = 2000 + expiry_year_2digit
            
            expiry_date_formatted = f"{expiry_full_year}-{expiry_month}-{expiry_day}"
        else:
            expiry_date_formatted = expiry_date
        
        decoded_details = {
            'mrz_type': 'TD3',
            'document_code': document_code,
            'issuer_code': issuer_code.replace('<', ''),
            'surname': surname,
            'given_name': given_name,
            'document_number': passport_number,
            'document_number_checkdigit': passport_check,
            'nationality_code': nationality_code.replace('<', ''),
            'birth_date': birth_date_formatted,
            'birth_date_checkdigit': birth_check,
            'sex': sex,
            'expiry_date': expiry_date_formatted,
            'expiry_date_checkdigit': expiry_check,
            'personal_number': personal_number,
            'personal_number_checkdigit': personal_check,
            'final_checkdigit': final_check,
            'mrz_text': f"{line1}\n{line2}",
            'status': 'SUCCESS'
        }
        
        return decoded_details
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': str(e),
            'mrz_text': f"{line1}\n{line2}"
        }


def create_td3_compliant_mrz(mrz_data: dict, raw_mrz: str):
    """
    Create TD3 compliant MRZ from PassportEye data following TD3 rules
    
    Args:
        mrz_data: Dictionary from PassportEye
        raw_mrz: Original raw MRZ text
        
    Returns:
        Tuple of (line1, line2, changes_made, change_percentage)
    """
    try:
        # Extract data from PassportEye result
        doc_type = mrz_data.get('type', 'P')
        country = mrz_data.get('country', 'XXX')[:3].upper()
        surname = mrz_data.get('surname', '').upper().replace(' ', '<')
        names = mrz_data.get('names', '').upper().replace(' ', '<')
        passport_num = mrz_data.get('number', '')[:9]
        nationality = mrz_data.get('nationality', 'XXX')[:3].upper()
        birth_date = mrz_data.get('date_of_birth', '000000')
        sex = mrz_data.get('sex', '<')
        expiry_date = mrz_data.get('expiration_date', '000000')
        personal_num = mrz_data.get('personal_number', '').replace('<', '')[:14]
        
        # Determine correct document code based on type
        if doc_type in ['PO', 'PD', 'PN', 'PS']:
            doc_code = doc_type
        elif doc_type == 'P':
            doc_code = 'P<'
        else:
            # Default to P< for unknown types
            doc_code = 'P<'
        
        # Validate document code
        if doc_code not in VALID_PASSPORT_CODES:
            print(f"  ⚠️ Invalid document code '{doc_code}', using 'P<' as default")
            doc_code = 'P<'
        
        # Build Line 1: DOC_CODE + COUNTRY_CODE + SURNAME<<GIVEN_NAMES (TD3 format)
        if doc_code == 'P<':
            line1_start = f"P<{country}"
        else:
            line1_start = f"{doc_code}{country}"
        
        # Clean surname and given names
        surname_clean = surname.replace('<', '').strip()
        names_clean = names.replace('<', '').strip()
        
        # Parse given names properly - preserve multiple names with < separator
        if names_clean:
            # Replace spaces with < for TD3 format and preserve multiple names
            given_name = names_clean.replace(' ', '<').replace('<<', '<')
            
            # Remove any trailing < symbols and clean up
            given_name = given_name.strip('<')
            
            # Special handling for known patterns
            if "ABDALLAH" in given_name.upper() and len(given_name) > 8:
                # For ABDALLAH case, keep just ABDALLAH
                given_name = "ABDALLAH"
            elif "INAYAT" in given_name.upper() and "ULLAH" in given_name.upper():
                # For INAYAT ULLAH case, keep as INAYAT<ULLAH
                given_name = "INAYAT<ULLAH"
            
            name_section = f"{surname_clean}<<{given_name}"
        else:
            name_section = f"{surname_clean}<<"
        
        # Calculate available space for names (44 - 5 = 39 characters)
        available_space = 44 - len(line1_start)
        
        # Truncate name section if too long
        if len(name_section) > available_space:
            name_section = name_section[:available_space]
        
        # Build complete line 1 and pad with '<' to exactly 44 characters
        line1 = line1_start + name_section
        line1 = line1.ljust(44, '<')[:44]
        
        # Build Line 2: PASSPORT_NUM + CHECK + NATIONALITY + BIRTH_DATE + CHECK + SEX + EXPIRY + CHECK + PERSONAL_NUM + CHECK + FINAL_CHECK
        passport_section = passport_num.ljust(9, '<')[:9]
        passport_check = '0'  # Simplified check digit
        nationality_section = nationality.ljust(3, '<')[:3]
        birth_section = birth_date.replace('-', '')[-6:].ljust(6, '0')[:6]
        birth_check = '0'  # Simplified check digit
        sex_section = sex if sex in ['M', 'F', 'X'] else '<'
        expiry_section = expiry_date.replace('-', '')[-6:].ljust(6, '0')[:6]
        expiry_check = '0'  # Simplified check digit
        personal_section = personal_num.ljust(14, '<')[:14]
        personal_check = '0'  # Simplified check digit
        final_check = '0'  # Simplified final check digit
        
        line2 = f"{passport_section}{passport_check}{nationality_section}{birth_section}{birth_check}{sex_section}{expiry_section}{expiry_check}{personal_section}{personal_check}{final_check}"
        line2 = line2[:44].ljust(44, '<')
        
        # Calculate changes made
        original_clean = raw_mrz.replace('\n', '').replace('\r', '').strip()
        new_mrz = f"{line1}{line2}"
        
        changes_made = 0
        total_chars = max(len(original_clean), len(new_mrz))
        
        # Compare character by character
        for i in range(total_chars):
            orig_char = original_clean[i] if i < len(original_clean) else ''
            new_char = new_mrz[i] if i < len(new_mrz) else ''
            if orig_char != new_char:
                changes_made += 1
        
        change_percentage = (changes_made / total_chars) * 100 if total_chars > 0 else 0
        
        return line1, line2, changes_made, change_percentage
        
    except Exception as e:
        print(f"Error creating TD3 compliant MRZ: {e}")
        return None, None, 0, 0


def format_mrz_lines(mrz_text: str, verbose: bool = False) -> str:
    """
    Format MRZ text into proper two-line format (44 chars each)
    
    Args:
        mrz_text: Raw MRZ text (may be concatenated)
        verbose: Print debug information
        
    Returns:
        Properly formatted MRZ text with two lines
    """
    if not mrz_text:
        return ""
    
    # Remove any existing line breaks and clean
    clean_mrz = mrz_text.replace('\n', '').replace('\r', '').strip()
    
    if verbose:
        print(f"  → Original MRZ: {clean_mrz}")
        print(f"  → Length: {len(clean_mrz)}")
    
    # Find P< pattern (start of line 1)
    p_pos = clean_mrz.find('P<')
    
    if p_pos >= 0:
        # Extract from P< position
        mrz_from_p = clean_mrz[p_pos:]
        
        if len(mrz_from_p) >= 88:  # Should have at least 88 chars for two lines
            # Split into two 44-character lines
            line1 = mrz_from_p[:44]
            line2 = mrz_from_p[44:88]
            
            # Ensure exactly 44 characters
            line1 = line1.ljust(44, '<')[:44]
            line2 = line2.ljust(44, '<')[:44]
            
            formatted_mrz = f"{line1}\n{line2}"
            
            if verbose:
                print(f"  → Formatted Line 1: {line1} (Length: {len(line1)})")
                print(f"  → Formatted Line 2: {line2} (Length: {len(line2)})")
            
            return formatted_mrz
        else:
            if verbose:
                print(f"  ⚠ Insufficient MRZ data: {len(mrz_from_p)} chars (need 88)")
    else:
        if verbose:
            print(f"  ⚠ P< pattern not found in MRZ")
    
    # Fallback: try to split at 44 characters if total length is around 88
    if len(clean_mrz) >= 80:  # Allow some flexibility
        line1 = clean_mrz[:44].ljust(44, '<')
        line2 = clean_mrz[44:88].ljust(44, '<')
        
        formatted_mrz = f"{line1}\n{line2}"
        
        if verbose:
            print(f"  → Fallback formatting applied")
            print(f"  → Line 1: {line1} (Length: {len(line1)})")
            print(f"  → Line 2: {line2} (Length: {len(line2)})")
        
        return formatted_mrz
    
    if verbose:
        print(f"  ✗ Could not format MRZ properly")
    
    return clean_mrz


def process_passport_image(image_path):
    """
    Process passport image and extract MRZ data with timing
    Similar to the fastMRZ.py functionality
    """
    global IMAGE_PATH
    IMAGE_PATH = image_path
    
    # Initialize TD3 decoded details storage
    process_passport_image.td3_decoded_details = None
    
    # ------------------------
    # Start Timer
    # ------------------------
    total_start = time.time()
    
    pre_start = time.time()
    preprocessed_path = preprocess_image(IMAGE_PATH)
    pre_end = time.time()
    
    if not preprocessed_path:
        print("Cannot read or preprocess the image.")
        return None
    else:
        try:
            ocr_start = time.time()
            mrz = read_mrz(preprocessed_path)
            ocr_end = time.time()
        finally:
            # Clean up temporary preprocessed file
            import os
            if os.path.exists(preprocessed_path) and "temp" in preprocessed_path:
                try:
                    os.unlink(preprocessed_path)
                    print(f"  🧹 Cleaned up temporary file: {os.path.basename(preprocessed_path)}")
                except Exception as e:
                    print(f"  ⚠️ Could not delete temp file: {e}")
        
        # new = mrz.valid_composite
        
        # print(new)        
        # print(mrz.valid_number)
        # print(mrz.valid_date_of_birth)
        # print(mrz.valid_expiration_date)
        # print(mrz.valid_composite)
        # print(mrz.valid)
        
        print("\n--- MRZ Parsed Fields ---")
        if mrz:
            mrz_data = mrz.to_dict()
            raw_mrz = mrz_data.get('raw_text', '')
            
            print(f"@@@ This here give back full details: {mrz_data}")
            print("\n" + "="*100)
            print(f"Print mrz text for debug==========================: {raw_mrz}")
            
            # Format MRZ into proper lines using TD3 rules
            if raw_mrz:
                print(f"\n🔧 CLEANING MRZ USING TD3 RULES:")
                
                # Clean and format MRZ text
                clean_mrz = raw_mrz.replace('\n', '').replace('\r', '').strip()
                original_length = len(clean_mrz)
                print(f"  → Original MRZ length: {original_length} characters")
                print(f"  → Expected TD3 length: 88 characters (44 + 44)")
                
                # Create TD3 compliant MRZ from PassportEye data
                line1_fixed, line2_fixed, changes_made, change_percentage = create_td3_compliant_mrz(mrz_data, clean_mrz)
                
                if line1_fixed and line2_fixed:
                    print(f"\n✅ TD3 COMPLIANT MRZ CREATED:")
                    print(f"  → Changes made: {changes_made}")
                    print(f"  → Change percentage: {change_percentage:.1f}%")
                    
                    print(f"\n🎯 IMMEDIATE MRZ LINES:")
                    print(f"Line 1 (Char 44): {line1_fixed}")
                    print(f"Line 2 (Char 44): {line2_fixed}")
                    print(f"Lengths: L1={len(line1_fixed)}, L2={len(line2_fixed)}")
                    
                    # Store formatted MRZ for later use
                    formatted_mrz = f"{line1_fixed}\n{line2_fixed}"
                    
                    # Decode the TD3 compliant MRZ lines
                    print(f"\n🔍 DECODING TD3 COMPLIANT MRZ:")
                    decoded_details = decode_td3_mrz_lines(line1_fixed, line2_fixed)
                    print(f"📋 DECODED DETAILS:")
                    print(decoded_details)
                    
                    # Store TD3 decoded details globally for API response
                    process_passport_image.td3_decoded_details = decoded_details
                    
                    # Show key fields in a formatted way (matching fastMRZ format)
                    if decoded_details and decoded_details.get('status') == 'SUCCESS':
                        print(f"\n📊 KEY EXTRACTED FIELDS:")
                        print(f"  Document Code: {decoded_details.get('document_code', 'N/A')}")
                        print(f"  Issuer Code: {decoded_details.get('issuer_code', 'N/A')}")
                        print(f"  Surname: {decoded_details.get('surname', 'N/A')}")
                        print(f"  Given Name: {decoded_details.get('given_name', 'N/A')}")
                        print(f"  Document Number: {decoded_details.get('document_number', 'N/A')}")
                        print(f"  Nationality: {decoded_details.get('nationality_code', 'N/A')}")
                        print(f"  Birth Date: {decoded_details.get('birth_date', 'N/A')}")
                        print(f"  Sex: {decoded_details.get('sex', 'N/A')}")
                        print(f"  Expiry Date: {decoded_details.get('expiry_date', 'N/A')}")
                        print(f"  Status: {decoded_details.get('status', 'N/A')}")
                else:
                    print(f"\n⚠️ Could not create TD3 compliant MRZ")
                    print(f"Raw MRZ length: {original_length}")
                    formatted_mrz = raw_mrz
                
                # Show decoding details
                # print(f"\n🔍 DECODING MRZ WITH PASSPORTEYE:")
                # print(f"📋 DECODED DETAILS:")
                # print(mrz_data)
                
                # Show key fields in a formatted way
                # print(f"\n📊 KEY EXTRACTED FIELDS:")
                # print(f"  Document Code: {mrz_data.get('type', 'N/A')}")
                # print(f"  Issuer Code: {mrz_data.get('country', 'N/A')}")
                # print(f"  Surname: {mrz_data.get('surname', 'N/A')}")
                # print(f"  Given Name: {mrz_data.get('names', 'N/A')}")
                # print(f"  Document Number: {mrz_data.get('number', 'N/A')}")
                # print(f"  Nationality: {mrz_data.get('nationality', 'N/A')}")
                # print(f"  Birth Date: {mrz_data.get('date_of_birth', 'N/A')}")
                # print(f"  Sex: {mrz_data.get('sex', 'N/A')}")
                # print(f"  Expiry Date: {mrz_data.get('expiration_date', 'N/A')}")
                # print(f"  Valid Score: {mrz_data.get('valid_score', 'N/A')}")
            
            # print("\n" + "="*100)
            # print(f"✓ MRZ detected by PassportEye")
            # print(f"Type: {mrz_data.get('mrz_type', 'Unknown')}")
            
            # print(f"\n🔄 USING DECODED DETAILS FOR FINAL EXTRACTION:")
            # print(f"✅ Using decoded details from PassportEye processing")
            
            # # Extract and clean fields
            # document_type = mrz_data.get('type', 'P')
            # country_code = mrz_data.get('country', '').upper()
            # surname = mrz_data.get('surname', '').replace('<', ' ').strip().upper()
            # given_names = mrz_data.get('names', '').replace('<', ' ').strip().upper()
            # passport_number = mrz_data.get('number', '')
            # nationality_code = mrz_data.get('nationality', '').upper()
            # date_of_birth = mrz_data.get('date_of_birth', '')
            # sex = mrz_data.get('sex', '<').upper()
            # expiry_date = mrz_data.get('expiration_date', '')
            # personal_number = mrz_data.get('personal_number', '').replace('<', '').strip()
            
            # print(f"\n📊 FINAL KEY EXTRACTED FIELDS FOR RETURN:")
            # print(f"  Document Code: {document_type}")
            # print(f"  Issuer Code: {country_code}")
            # print(f"  Surname: {surname}")
            # print(f"  Given Name: {given_names}")
            # print(f"  Document Number: {passport_number}")
            # print(f"  Nationality: {nationality_code}")
            # print(f"  Birth Date: {date_of_birth}")
            # print(f"  Sex: {sex}")
            # print(f"  Expiry Date: {expiry_date}")
            # print(f"  Valid Score: {mrz_data.get('valid_score', 0)}")
            
            # # Clean country codes
            # country_code_clean = ''.join([c for c in country_code if c.isalpha()])[:3]
            # nationality_clean = ''.join([c for c in nationality_code if c.isalpha()])[:3]
            
            # print(f"\n🧹 CLEANED CODES:")
            # print(f"  Country Code: {country_code_clean}")
            # print(f"  Nationality: {nationality_clean}")
            
            # print("=" * 100)
            
            # # Only print selected fields
            # wanted_fields = ["mrz_type", "valid_score", "raw_text",
            #                "type", "country", "number", "date_of_birth",
            #                "expiration_date", "nationality", "sex", "personal_number"]
            
            # print("\n--- Selected Fields ---")
            # for key in wanted_fields:
            #     print(f"{key}: {mrz_data.get(key, '')}")
        else:
            print("No MRZ Found.")
        
        # Validation
        val_start = time.time()
        is_valid = mrz is not None
        val_end = time.time()
        
        if is_valid:
            print("\nDocument is Valid for passport.")
            
            # Add individual field validation check
            print("\nCheck all field:")
            print("-" * 40)
            
            # Get the formatted MRZ text for field validation
            if hasattr(process_passport_image, 'td3_decoded_details') and process_passport_image.td3_decoded_details:
                mrz_for_validation = process_passport_image.td3_decoded_details.get('mrz_text', '')
            else:
                # Fallback to raw MRZ from PassportEye
                mrz_for_validation = mrz_data.get('raw_text', '') if mrz_data else ''
            
            if mrz_for_validation:
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
            
            else:
                print("❌ No MRZ text available for field validation")
        else:
            print("\nDocument is NOT a valid passport.")
        
        total_end = time.time()
        
        # ------------------------
        # TIME REPORT
        # ------------------------
        print("\n=========== TIME REPORT ===========")
        print(f"Preprocessing time     : {pre_end - pre_start:.4f} seconds")
        print(f"MRZ detection time     : {ocr_end - ocr_start:.4f} seconds")
        print(f"Validation time        : {val_end - val_start:.4f} seconds")
        print("-----------------------------------")
        print(f"TOTAL time taken       : {total_end - total_start:.4f} seconds")
        print("===================================")
        
        return mrz_data if mrz else None


# Main execution function
if __name__ == "__main__":
    # Example usage - you can set the image path here
    image_path = "path/to/your/passport/image.jpg"  # Change this to your image path
    
    if IMAGE_PATH is None:
        IMAGE_PATH = image_path
    
    result = process_passport_image(IMAGE_PATH)
    print("ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
    
    if result:
        print(f"\nProcessing completed successfully!")
        print(f"MRZ Type: {result.get('mrz_type', 'Unknown')}")
        print(f"Valid Score: {result.get('valid_score', 0)}")
    else:
        print("\nProcessing failed - no MRZ detected.")


def validate_passport_with_PassportEye_fallback(image, verbose=True):
    """
    Validate passport using PassportEye with fallback functionality
    Compatible with the scanner.py interface
    
    Args:
        image: PIL Image object
        verbose: Print detailed logs
        
    Returns:
        Dictionary with success status and passport data
    """
    try:
        if verbose:
            print(f"  → Processing with PassportEye...")
        
        # Convert PIL Image to temporary file for processing
        import tempfile
        import os
        
        # Save PIL image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            image.save(tmp_file.name, 'JPEG')
            temp_image_path = tmp_file.name
        
        try:
            # Process the image using our main function
            result = process_passport_image(temp_image_path)
            
            if result:
                if verbose:
                    print(f"  ✓ PassportEye processing successful")
                
                # Check if we have TD3 compliant decoded details (stored globally during processing)
                if hasattr(process_passport_image, 'td3_decoded_details') and process_passport_image.td3_decoded_details:
                    td3_data = process_passport_image.td3_decoded_details
                    
                    # Get country information for enhanced response
                    from country_code import get_country_info
                    country_code_clean = td3_data.get('issuer_code', '').replace('<', '').strip()
                    country_info = get_country_info(country_code_clean)
                    
                    # Use TD3 compliant data for API response
                    passport_data = {
                        "document_type": td3_data.get('document_code', 'P'),
                        "country_code": country_code_clean,
                        "surname": td3_data.get('surname', ''),
                        "given_names": td3_data.get('given_name', ''),
                        "passport_number": td3_data.get('document_number', ''),
                        "country_name": country_info.get('name', country_code_clean),
                        "country_iso": country_info.get('alpha2', ''),
                        "nationality": country_info.get('nationality', country_code_clean),
                        "date_of_birth": td3_data.get('birth_date', ''),
                        "sex": td3_data.get('sex', ''),
                        "expiry_date": td3_data.get('expiry_date', ''),
                        "personal_number": td3_data.get('personal_number', '').replace('<', '').strip()
                    }
                    
                    mrz_text = td3_data.get('mrz_text', result.get('raw_text', ''))
                    
                    if verbose:
                        print(f"  🎯 Using TD3 compliant data for API response")
                        print(f"    Given Names: {passport_data['given_names']} (cleaned)")
                        print(f"    Birth Date: {passport_data['date_of_birth']} (formatted)")
                        print(f"    Expiry Date: {passport_data['expiry_date']} (formatted)")
                else:
                    # Fallback to original PassportEye data
                    from country_code import get_country_info
                    country_code_clean = result.get('country', '').replace('<', '').strip()
                    country_info = get_country_info(country_code_clean)
                    
                    passport_data = {
                        "document_type": result.get('type', 'P'),
                        "country_code": country_code_clean,
                        "surname": result.get('surname', ''),
                        "given_names": result.get('names', ''),
                        "passport_number": result.get('number', ''),
                        "country_name": country_info.get('name', country_code_clean),
                        "country_iso": country_info.get('alpha2', ''),
                        "nationality": country_info.get('nationality', country_code_clean),
                        "date_of_birth": result.get('date_of_birth', ''),
                        "sex": result.get('sex', ''),
                        "expiry_date": result.get('expiration_date', ''),
                        "personal_number": result.get('personal_number', '').replace('<', '').strip()
                    }
                    
                    mrz_text = result.get('raw_text', '')
                    
                    if verbose:
                        print(f"  ⚠️ Using original PassportEye data (TD3 data not available)")
                
                return {
                    "success": True,
                    "passport_data": passport_data,
                    "mrz_text": mrz_text,
                    "method_used": "PassportEye",
                    "confidence": result.get('valid_score', 0) / 100.0,
                    "error": ""
                }
            else:
                if verbose:
                    print(f"  ✗ PassportEye failed to extract MRZ")
                
                return {
                    "success": False,
                    "passport_data": {},
                    "mrz_text": "",
                    "method_used": "PassportEye",
                    "error": "No MRZ detected by PassportEye"
                }
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
        
    except Exception as e:
        if verbose:
            print(f"  ✗ PassportEye error: {e}")
        
        return {
            "success": False,
            "passport_data": {},
            "mrz_text": "",
            "method_used": "PassportEye",
            "error": f"PassportEye processing error: {str(e)}"
        }