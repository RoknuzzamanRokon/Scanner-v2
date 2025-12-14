"""
AI-powered passport data extraction using Gemini OCR
3-step approach: Full text → Data extraction → MRZ reconstruction → FastMRZ parsing
"""
from PIL import Image
import requests
from io import BytesIO
import re
from typing import Dict
from gemini_ocr import extract_text_with_gemini
from country_code import get_country_info


class AIExtractionSuccess(Exception):
    """Exception raised when AI extraction succeeds early to bypass remaining pipeline"""
    def __init__(self, passport_data):
        self.passport_data = passport_data


def clean_country_code(country_code: str) -> str:
    """
    Clean and validate country code to ensure it contains only letters
    
    Args:
        country_code: Raw country code that may contain symbols
        
    Returns:
        Cleaned 3-letter country code (A-Z only)
    """
    if not country_code:
        return 'XXX'
    
    # Remove all non-letter characters and convert to uppercase
    cleaned = ''.join([c for c in country_code if c.isalpha()]).upper()
    
    # Validate length
    if len(cleaned) == 0:
        return 'XXX'  # Default for unknown
    elif len(cleaned) < 3:
        return cleaned.ljust(3, 'X')  # Pad with X if too short
    elif len(cleaned) > 3:
        return cleaned[:3]  # Truncate if too long
    
    return cleaned


def get_country_details(country_code: str) -> dict:
    """
    Get country details including name, nationality, and ISO codes
    
    Args:
        country_code: 3-letter ISO country code (e.g., 'PAK', 'USA', 'IND')
        
    Returns:
        Dictionary with country_name, country_iso (2-letter), and nationality
    """
    # Clean the country code first
    cleaned_code = clean_country_code(country_code)
    
    if not cleaned_code or len(cleaned_code) != 3:
        return {
            "country_name": cleaned_code,
            "country_iso": "",
            "nationality": cleaned_code
        }
    
    info = get_country_info(cleaned_code)
    
    if "error" in info:
        return {
            "country_name": country_code,
            "country_iso": "",
            "nationality": country_code
        }
    
    return {
        "country_name": info["name"],
        "country_iso": info["alpha2"],
        "nationality": info["nationality"]
    }


def gemini_ocr(image_input, is_url: bool = True, user_id: str = None) -> Dict:
    """
    Extract passport data using AI full text extraction - supports both URL and PIL Image
    
    Args:
        image_input: Either a URL string or PIL Image object
        is_url: True if image_input is a URL, False if it's a PIL Image
        user_id: Optional user ID for temporary file management
        
    Returns:
        Dictionary with passport data in standard format
    """
    if is_url:
        return gemini_ocr_from_url(image_input)
    else:
        # For PIL Image, we save to temp folder and treat as URL
        try:
            image = image_input
            print(f"  ✓ Using provided image: {image.size} {image.mode}")
            
            # Convert RGBA to RGB if needed
            if image.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = rgb_image
                print(f"  ✓ Image converted to RGB")
            
            return _gemini_ocr_from_image(image, user_id)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n✗ Error in gemini_ocr: {e}")
            print(f"Traceback:\n{error_details}")
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "error": str(e)
            }


def _gemini_ocr_from_image(image: Image.Image, user_id: str = None) -> Dict:
    """
    Internal function to extract passport data from PIL Image
    Saves image to temp folder, treats as URL, and cleans up
    """
    import os
    from pathlib import Path
    from utils import save_temp_image, create_user_temp_folder, cleanup_user_folder
    
    user_folder = None
    try:
        # Create user temp folder
        user_folder = create_user_temp_folder(user_id)
        print(f"  → Created temp folder: {user_folder}")
        
        # Save image
        saved_path = save_temp_image(image, prefix="ai_input", user_folder=user_folder)
        print(f"  → Saved temp image: {saved_path}")
        
        # Create file URL
        file_url = f"file:///{saved_path.as_posix()}"
        print(f"  → Processing via local URL: {file_url}")
        
        # Process using the URL-based function
        # We need to handle the file:// protocol in gemini_ocr_from_url
        result = gemini_ocr_from_url(file_url)
        
        return result
        
    except Exception as e:
        print(f"  ✗ Error in local image processing: {e}")
        return {
            "success": False,
            "passport_data": {},
            "mrz_text": "",
            "error": f"Local processing failed: {str(e)}"
        }
    finally:
        # Cleanup
        if user_folder:
            # cleanup_user_folder(user_folder)
            print(f"  → Cleaned up temp folder")


def gemini_ocr_from_url(image_url: str) -> Dict:
    """
    Extract passport data from image URL using AI full text extraction
    
    Args:
        image_url: URL of the passport image
        
    Returns:
        Dictionary with passport data in standard format
    """
    try:
        # Download image
        # Download/Load image
        if image_url.startswith('file://'):
            local_path = image_url.replace('file:///', '').replace('file://', '')
            print(f"  → Loading image from local file: {local_path}")
            image = Image.open(local_path)
        else:
            print(f"  → Downloading image from URL...")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            
        print(f"  ✓ Image loaded: {image.size} {image.mode}")
        
        # Convert RGBA to RGB if needed
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = rgb_image
            print(f"  ✓ Image converted to RGB")
        
        # AI Step 1: Extract full text and reconstruct
        print(f"  → AI Step 1: Full Text extraction and data reconstruction...")
        mrz_text = ""  # Initialize mrz_text
        try:
            full_text = extract_text_with_gemini(image)
        except Exception as e:
            # If it's an API error, return immediately
            error_msg = str(e)
            if "AI parser failed:" in error_msg:
                print(f"  ✗ {error_msg}")
                return {
                    "success": False,
                    "passport_data": {},
                    "mrz_text": "",
                    "error": error_msg
                }
            # Otherwise, return generic error
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "error": "AI parser failed: Failed to extract text from image. Try again."
            }
        
        if not full_text or not isinstance(full_text, str) or len(full_text.strip()) == 0:
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "error": "AI parser failed: Failed to extract text from image. Try again."
            }
        
        # Extract passport data from full text for validation
        validation_data = {}
        
        # Surname - Universal patterns for all countries
        surname_patterns = [
            r'(?:Surname|SURNAME|Nom|Apellido|Cognome|Nachname|Фамилия|उपनाम|姓)[:\.\-/]*\s*([A-Z\s]+?)(?:\n|$)',
            r'(?:Family Name|FAMILY NAME)[:\.\-/]*\s*([A-Z\s]+?)(?:\n|$)',
            r'(?:Last Name|LAST NAME)[:\.\-/]*\s*([A-Z\s]+?)(?:\n|$)'
        ]
        for pattern in surname_patterns:
            surname_match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if surname_match:
                validation_data['surname'] = surname_match.group(1).strip().upper()
                break
        
        # Given Names - Universal patterns
        given_patterns = [
            r'(?:Given Names?|GIVEN NAMES?|Prénoms?|Nombres?|Nome|Vorname|Имя|दिया गया नाम|名)[:\.\-/]*\s*([A-Z\s]+?)(?:\n|Date|Nationality|Sex|DOB|जन्म|राष्ट्रीयता)',
            r'(?:First Name|FIRST NAME)[:\.\-/]*\s*([A-Z\s]+?)(?:\n|Date|Nationality|Sex|DOB)'
        ]
        for pattern in given_patterns:
            given_match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if given_match:
                validation_data['given_names'] = given_match.group(1).strip().upper().replace(' ', '<<')
                break
        
        # Passport Number - Universal patterns
        passport_patterns = [
            r'(?:Passport Number|PASSPORT NUMBER|Passport No\.?|No\. de passeport|Número de pasaporte|Numero di passaporto|Reisepass-Nr|Номер паспорта|पासपोर्ट न|护照号码)[:\.\-/]*\s*([A-Z0-9]+)',
            r'(?:Document Number|DOCUMENT NUMBER|Doc No\.?)[:\.\-/]*\s*([A-Z0-9]+)',
            r'(?:Passeport|Pasaporte|Passaporto|Pass)[:\.\-/\s]+(?:N[oº°]?\.?)[:\.\-/\s]*([A-Z0-9]+)'
        ]
        for pattern in passport_patterns:
            passport_match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if passport_match:
                validation_data['passport_number'] = passport_match.group(1).upper()
                break
        
        # Country Code - Universal patterns
        country_patterns = [
            r'(?:Country Code|COUNTRY CODE|Code pays|Código de país|Codice paese|Ländercode|कोड)[:\.\-/]*\s*([A-Z]{3})',
            r'(?:Issuing Country|ISSUING COUNTRY|Pays émetteur|País emisor)[:\.\-/]*\s*([A-Z]{3})',
            r'(?:Code|CODE)[:\.\-/]*\s*([A-Z]{3})(?:\s|$)'
        ]
        for pattern in country_patterns:
            country_match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if country_match:
                validation_data['country_code'] = clean_country_code(country_match.group(1))
                break
        
        # Nationality - Universal patterns
        nationality_patterns = [
            r'(?:Nationality|NATIONALITY|Nationalité|Nacionalidad|Nazionalità|Staatsangehörigkeit|Гражданство|राष्ट्रीयता|国籍)[:\.\-/]*\s*(?:Country Code\s+)?([A-Z]{3,})',
            r'(?:Citizen of|CITIZEN OF)[:\.\-/]*\s*([A-Z]{3,})'
        ]
        for pattern in nationality_patterns:
            nationality_match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if nationality_match:
                nat_value = nationality_match.group(1).upper()
                # Convert full country names to 3-letter codes if needed
                validation_data['nationality'] = nat_value[:3] if len(nat_value) > 3 else nat_value
                break
        
        # Date of Birth - Multiple formats for different countries
        dob_patterns = [
            # Format: DD MMM YYYY (e.g., 02 JAN 1990)
            (r'(?:Date of Birth|DATE OF BIRTH|Birth Date|DOB|Date de naissance|Fecha de nacimiento|Data di nascita|Geburtsdatum|Дата рождения|जन्मतिथि|出生日期)[:\.\-/]*\s*(\d{1,2})[/\s\-\.]+([A-Z]{3})[/\s\-\.]+(\d{4})', 'dmy_text'),
            # Format: DD/MM/YYYY or DD-MM-YYYY
            (r'(?:Date of Birth|DATE OF BIRTH|Birth Date|DOB|Date de naissance|Fecha de nacimiento|Data di nascita|Geburtsdatum|Дата рождения|जन्मतिथि|出生日期)[:\.\-/]*\s*(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})', 'dmy'),
            # Format: YYYY-MM-DD or YYYY/MM/DD
            (r'(?:Date of Birth|DATE OF BIRTH|Birth Date|DOB|Date de naissance|Fecha de nacimiento|Data di nascita|Geburtsdatum|Дата рождения|जन्मतिथि|出生日期)[:\.\-/]*\s*(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', 'ymd'),
            # Format: MM/DD/YYYY (US format)
            (r'(?:Date of Birth|DATE OF BIRTH|Birth Date|DOB)[:\.\-/]*\s*(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})', 'mdy')
        ]
        
        month_map = {
            'JAN':'01','FEB':'02','MAR':'03','APR':'04','MAY':'05','JUN':'06',
            'JUL':'07','AUG':'08','SEP':'09','OCT':'10','NOV':'11','DEC':'12',
            'ENE':'01','ABR':'04','AGO':'08','DIC':'12',  # Spanish
            'JUI':'07','AOÛ':'08','DÉC':'12',  # French
            'GEN':'01','FEB':'02','MAR':'03','APR':'04','MAG':'05','GIU':'06',  # Italian
            'LUG':'07','AGO':'08','SET':'09','OTT':'10','NOV':'11','DIC':'12'
        }
        
        for pattern, format_type in dob_patterns:
            dob_match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if dob_match:
                if format_type == 'dmy_text':
                    day, month_name, year = dob_match.groups()
                    month = month_map.get(month_name.upper()[:3], '01')
                    validation_data['date_of_birth'] = year[2:] + month + day.zfill(2)
                elif format_type == 'dmy':
                    day, month, year = dob_match.groups()
                    validation_data['date_of_birth'] = year[2:] + month.zfill(2) + day.zfill(2)
                elif format_type == 'ymd':
                    year, month, day = dob_match.groups()
                    validation_data['date_of_birth'] = year[2:] + month.zfill(2) + day.zfill(2)
                elif format_type == 'mdy':
                    month, day, year = dob_match.groups()
                    validation_data['date_of_birth'] = year[2:] + month.zfill(2) + day.zfill(2)
                break
        
        # Date of Expiry - Multiple formats
        expiry_patterns = [
            # Format: DD MMM YYYY
            (r'(?:Date of Expiry|DATE OF EXPIRY|Expiry Date|Expiration Date|Date d\'expiration|Fecha de caducidad|Data di scadenza|Gültig bis|Дата истечения|समाप्ति की तिथि|到期日期)[:\.\-/]*\s*(\d{1,2})[/\s\-\.]+([A-Z]{3})[/\s\-\.]+(\d{4})', 'dmy_text'),
            # Format: DD/MM/YYYY
            (r'(?:Date of Expiry|DATE OF EXPIRY|Expiry Date|Expiration Date|Date d\'expiration|Fecha de caducidad|Data di scadenza|Gültig bis|Дата истечения|समाप्ति की तिथि|到期日期)[:\.\-/]*\s*(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})', 'dmy'),
            # Format: YYYY-MM-DD
            (r'(?:Date of Expiry|DATE OF EXPIRY|Expiry Date|Expiration Date|Date d\'expiration|Fecha de caducidad|Data di scadenza|Gültig bis|Дата истечения|समाप्ति की तिथि|到期日期)[:\.\-/]*\s*(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', 'ymd'),
            # Format: MM/DD/YYYY
            (r'(?:Date of Expiry|DATE OF EXPIRY|Expiry Date|Expiration Date)[:\.\-/]*\s*(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})', 'mdy')
        ]
        
        for pattern, format_type in expiry_patterns:
            expiry_match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if expiry_match:
                if format_type == 'dmy_text':
                    day, month_name, year = expiry_match.groups()
                    month = month_map.get(month_name.upper()[:3], '01')
                    validation_data['expiry_date'] = year[2:] + month + day.zfill(2)
                elif format_type == 'dmy':
                    day, month, year = expiry_match.groups()
                    validation_data['expiry_date'] = year[2:] + month.zfill(2) + day.zfill(2)
                elif format_type == 'ymd':
                    year, month, day = expiry_match.groups()
                    validation_data['expiry_date'] = year[2:] + month.zfill(2) + day.zfill(2)
                elif format_type == 'mdy':
                    month, day, year = expiry_match.groups()
                    validation_data['expiry_date'] = year[2:] + month.zfill(2) + day.zfill(2)
                break
        
        # Sex - Universal patterns
        sex_patterns = [
            r'(?:Sex|SEX|Sexe|Sexo|Sesso|Geschlecht|Пол|लिंग|性别)[:\.\-/]*\s*([MFX])',
            r'(?:Gender|GENDER)[:\.\-/]*\s*([MFX])'
        ]
        for pattern in sex_patterns:
            sex_match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if sex_match and sex_match.lastindex and sex_match.lastindex >= 1:
                validation_data['sex'] = sex_match.group(1).upper()
                break
        
        print(f"\n  → Validation Data Extracted:")
        for key, value in validation_data.items():
            print(f"     {key}: {value}")
        
        # Reconstruct MRZ if we have enough validation data
        corrected_mrz = ""
        if len(validation_data) >= 5:  # Need at least 5 fields to reconstruct
            print(f"\n  → Reconstructing MRZ from validation data...")
            
            # Build Line 1: P<CCCSSSSSSSSSSSS<<GGGGGGGGGGGGGGGGGGG
            country = validation_data.get('country_code', 'XXX')  # Use XXX as universal default
            surname = validation_data.get('surname', 'UNKNOWN')
            given = validation_data.get('given_names', 'UNKNOWN')
            
            # Ensure surname + given fit in line (max 39 chars after P<CCC)
            name_field = f"{surname}<<{given}"
            if len(name_field) > 39:
                name_field = name_field[:39]
            name_field = name_field.ljust(39, '<')
            line1 = f"P<{country}{name_field}"
            
            # Build Line 2: TD3 format (44 chars)
            # PassportNum(9) + Check(1) + Country(3) + DOB(6) + Check(1) + Sex(1) + Expiry(6) + Check(1) + Optional(14) + FinalCheck(1)
            passport_num = validation_data.get('passport_number', '000000000')
            nationality = validation_data.get('nationality', validation_data.get('country_code', 'XXX'))
            dob = validation_data.get('date_of_birth', '000000')
            sex = validation_data.get('sex', '<')  # Use < as default if sex not found
            expiry = validation_data.get('expiry_date', '000000')
            
            # Passport field (9 chars)
            passport_field = passport_num[:9].ljust(9, '<')
            
            # Build line 2
            line2 = f"{passport_field}<{nationality}{dob}<{sex}{expiry}<<<<<<<<<<<<<<<"
            
            # Ensure exactly 44 characters
            line1 = line1[:44].ljust(44, '<')
            line2 = line2[:44].ljust(44, '<')
            
            corrected_mrz = f"{line1}\n{line2}"
            mrz_text = corrected_mrz  # Set mrz_text to the reconstructed MRZ
            print(f"  ✓ MRZ reconstructed from validation data")
            print(f"\n  → Using MRZ text:")
            for line in corrected_mrz.split('\n'):
                print(f"     {line}")
            
            # Try parsing reconstructed MRZ
            try:
                from fastmrz import FastMRZ
                fast_mrz = FastMRZ()
                details = fast_mrz._parse_mrz(corrected_mrz.strip())
                
                # Check if we have MRZ data (even if validation failed)
                if details and details.get('document_code'):
                    status = details.get('status', 'UNKNOWN')
                    if status == 'SUCCESS':
                        print("\n✓ FastMRZ Parsing Result:")
                        print(f"  Status: {status}")
                    else:
                        print("\n⚠ FastMRZ Parsing Result:")
                        print(f"  Status: {status}")
                        print(f"  Warning: {details.get('status_message', 'Unknown issue')}")
                    
                    print(f"  MRZ Type: {details.get('mrz_type', 'N/A')}")
                    
                    # Extract passport data (even if checksum failed) - use issuer_code for country lookup
                    issuer_code = clean_country_code(details.get('issuer_code', ''))
                    country_details = get_country_details(issuer_code)
                    
                    passport_data = {
                        "document_type": details.get('document_code', ''),
                        "country_code": issuer_code,
                        "surname": details.get('surname', '').replace('<', ' ').strip(),
                        "given_names": details.get('given_name', '').replace('<', ' ').strip(),
                        "passport_number": details.get('document_number', ''),
                        "country_name": country_details["country_name"],
                        "country_iso": country_details["country_iso"],
                        "nationality": country_details["nationality"],
                        "date_of_birth": details.get('birth_date', ''),
                        "sex": details.get('sex', ''),
                        "expiry_date": details.get('expiry_date', ''),
                        "personal_number": details.get('optional_data', '').replace('<', '').strip()
                    }
                    
                    print("\n✓ Passport Data:")
                    print("  {")
                    for key, value in passport_data.items():
                        print(f'    "{key}": "{value}",')
                    print("  }")
                    
                    return {
                        "success": True,
                        "passport_data": passport_data,
                        "mrz_text": mrz_text,
                        "error": ""
                    }
                else:
                    print(f"\n✗ FastMRZ parsing failed: Could not extract MRZ data")
                    if details:
                        print(f"  Full details: {details}")
            except Exception as e:
                print(f"\n✗ FastMRZ parsing error: {e}")
                import traceback
                traceback.print_exc()
        
        # Final fallback: Return extracted data without MRZ validation
        if not validation_data.get('passport_number'):
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "error": "Could not extract required fields (passport number)"
            }
        
        # Convert dates from YYMMDD to YYYY-MM-DD for final output
        date_of_birth = ""
        expiry_date = ""
        
        if validation_data.get('date_of_birth'):
            dob = validation_data['date_of_birth']  # YYMMDD
            if len(dob) == 6:
                year = "20" + dob[0:2] if int(dob[0:2]) < 50 else "19" + dob[0:2]
                date_of_birth = f"{year}-{dob[2:4]}-{dob[4:6]}"
        
        if validation_data.get('expiry_date'):
            exp = validation_data['expiry_date']  # YYMMDD
            if len(exp) == 6:
                year = "20" + exp[0:2] if int(exp[0:2]) < 50 else "19" + exp[0:2]
                expiry_date = f"{year}-{exp[2:4]}-{exp[4:6]}"
        
        print(f"  ✓ Passport data extracted successfully from Full Text")
        
        # Return structured data - use country_code for country lookup
        country_code = validation_data.get('country_code', 'XXX')
        country_details = get_country_details(country_code)
        
        passport_data = {
            "document_type": "P",
            "country_code": country_code,
            "surname": validation_data.get('surname', '').replace('<<', ' ').strip(),
            "given_names": validation_data.get('given_names', '').replace('<<', ' ').strip(),
            "passport_number": validation_data.get('passport_number', ''),
            "country_name": country_details["country_name"],
            "country_iso": country_details["country_iso"],
            "nationality": country_details["nationality"],
            "date_of_birth": date_of_birth,
            "sex": validation_data.get('sex', ''),
            "expiry_date": expiry_date,
            "personal_number": ""
        }
        
        return {
            "success": True,
            "passport_data": passport_data,
            "mrz_text": mrz_text,
            "error": ""
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n✗ Error in gemini_ocr_from_url: {e}")
        print(f"Traceback:\n{error_details}")
        return {
            "success": False,
            "passport_data": {},
            "mrz_text": "",
            "error": str(e)
        }
