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
    
    # Fix name separators in MRZ line 1
    # Proper format: P<COUNTRY_CODE + SURNAME + << + GIVEN_NAME1 + < + GIVEN_NAME2 + <<<...
    def fix_name_separator(mrz_line):
        """Fix name separators in MRZ line 1 to follow proper TD3 format"""
        if not mrz_line.startswith('P<'):
            return mrz_line
            
        # Only process if this looks like line 1 (contains letters after P<)
        if len(mrz_line) < 44:
            return mrz_line
            
        # Extract the first 44 characters (line 1)
        line1 = mrz_line[:44]
        rest = mrz_line[44:] if len(mrz_line) > 44 else ""
        
        # Find country code (3 chars after P<)
        country_code = line1[2:5]
        name_section = line1[5:]
        
        import re
        
        # Extract all name components (letters only, ignoring < separators)
        name_parts = re.findall(r'[A-Z]+', name_section)
        
        if len(name_parts) >= 1:
            # We have at least a surname
            surname = name_parts[0]
            given_names = name_parts[1:] if len(name_parts) > 1 else []
            
            # Build the correct name section format:
            # SURNAME << GIVEN_NAME1 < GIVEN_NAME2 < ... <<<<<<<<<
            if given_names:
                # Format: SURNAME << GIVEN_NAME1 < GIVEN_NAME2 < ...
                formatted_names = surname + '<<' + '<'.join(given_names)
            else:
                # Only surname: SURNAME <<
                formatted_names = surname + '<<'
            
            # Calculate remaining space for padding
            remaining_space = 39 - len(formatted_names)  # 44 - 5 (P< + country) = 39
            
            if remaining_space > 0:
                # Add padding with <
                formatted_names += '<' * remaining_space
            elif remaining_space < 0:
                # Names too long, truncate from the end
                formatted_names = formatted_names[:39]
            
            # Reconstruct line 1
            fixed_line1 = f"P<{country_code}{formatted_names}"
            
            # Reconstruct the full MRZ
            fixed_mrz = fixed_line1 + rest
            
            if verbose and fixed_line1 != line1:
                print(f"  → Fixed name separators:")
                print(f"    Before: {line1}")
                print(f"    After:  {fixed_line1}")
                print(f"    Names:  {surname} + {given_names}")
            
            return fixed_mrz
        
        return mrz_line
    
    # Apply the fix
    clean_mrz = fix_name_separator(clean_mrz)
    
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
            print(f"@@@ This here give back full details: {details}")
            print("\n" + "="*100)
            raw_mrz = details.get('mrz_text', '')
            print(f"Print mrz text for debug==========================: {raw_mrz}")
            
            # Initialize decoded_details outside the if block
            decoded_details = None
            formatted_mrz_text = ""
            
            # Here get 2 line data follow 'mrz_text'.
            # Separate from follow 'p<' for line one.
            if raw_mrz:
                # Quick format for immediate display
                formatted = format_mrz_lines(raw_mrz, verbose=False)
                print("--------------89898989454fdafadfaadfddfa------------", formatted)
                if formatted and '\n' in formatted:
                    lines = formatted.split('\n')
                    if len(lines) >= 2:
                        print(f"\n🎯 IMMEDIATE MRZ LINES:")
                        print(f"Line 1 (Char 44): {lines[0]}")
                        print(f"Line 2 (Char 44): {lines[1]}")
                        print(f"Lengths: L1={len(lines[0])}, L2={len(lines[1])}")
                        
                        # Store formatted MRZ text for later use
                        formatted_mrz_text = f"{lines[0]}\n{lines[1]}"
                        
                        # print("++++++++++++++++++++++",formatted_mrz_text)
                        
                        # Decode MRZ using FastMRZ _parse_mrz method
                        print(f"\n🔍 DECODING MRZ WITH FASTMRZ:")
                        try:
                            decode_fast_mrz = FastMRZ()
                            decoded_details = decode_fast_mrz._parse_mrz(formatted_mrz_text)
                            print(f"📋 DECODED DETAILS:")
                            print(decoded_details)
                            
                            # Show key fields in a formatted way
                            if decoded_details:
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
                        except Exception as e:
                            print(f"❌ Error decoding MRZ: {e}")
                            decoded_details = None
            print("\n" + "="*100)
            
            
            if not decoded_details:
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
                print(f"    Type: {decoded_details.get('mrz_type', 'Unknown')}")
            
            # Check for required fields
            if not decoded_details.get('document_number'):
                if verbose:
                    print(f"  ✗ No document number found")
                return {
                    "success": False,
                    "passport_data": {},
                    "mrz_text": "",
                    "method_used": "FastMRZ",
                    "error": "No document number extracted"
                }
            
            # Use decoded details from the first decoding section
            print(f"\n🔄 USING DECODED DETAILS FOR FINAL EXTRACTION:")
            
            if decoded_details and decoded_details.get('status') == 'SUCCESS':
                print(f"✅ Using decoded details from first decoding section")
                
                # Extract fields directly from decoded_details
                document_type = decoded_details.get('document_code', 'P')
                country_code = decoded_details.get('issuer_code', '').upper()
                
                # Manual name parsing from MRZ to ensure correct extraction
                def parse_names_from_mrz(mrz_text):
                    """Parse surname and given names correctly from MRZ line 1"""
                    try:
                        lines = mrz_text.strip().split('\n')
                        if len(lines) < 1:
                            return "", ""
                        
                        line1 = lines[0]
                        if len(line1) < 5:
                            return "", ""
                        
                        # Extract name section (after P< + 3-char country code)
                        name_section = line1[5:]
                        
                        # Split by << to separate surname from given names
                        if '<<' in name_section:
                            parts = name_section.split('<<', 1)
                            surname_raw = parts[0]
                            given_names_raw = parts[1] if len(parts) > 1 else ""
                            
                            # Clean surname
                            surname = surname_raw.replace('<', ' ').strip()
                            
                            # Clean given names - replace < with space, remove trailing <
                            given_names = given_names_raw.rstrip('<').replace('<', ' ').strip()
                            
                            return surname, given_names
                        else:
                            # No << separator found, treat all as surname
                            surname = name_section.replace('<', ' ').strip()
                            return surname, ""
                    except Exception:
                        return "", ""
                
                # Use manual parsing for names
                surname, given_names = parse_names_from_mrz(formatted_mrz_text)
                
                # Fallback to library extraction if manual parsing fails
                if not surname:
                    surname = decoded_details.get('surname', '').replace('<', ' ').strip().upper()
                if not given_names:
                    given_names = decoded_details.get('given_name', '').replace('<', ' ').strip().upper()
                passport_number = decoded_details.get('document_number', '')
                nationality_code = decoded_details.get('nationality_code', '').upper()
                date_of_birth = decoded_details.get('birth_date', '')  # Already in YYYY-MM-DD format
                sex = decoded_details.get('sex', '<').upper()
                expiry_date = decoded_details.get('expiry_date', '')  # Already in YYYY-MM-DD format
                personal_number = decoded_details.get('optional_data', '').replace('<', '').strip()
                
                # Use the formatted MRZ text from decoding
                mrz_text = formatted_mrz_text
                
                print(f"\n📊 FINAL KEY EXTRACTED FIELDS FOR RETURN:")
                print(f"  Document Code: {document_type}")
                print(f"  Issuer Code: {country_code}")
                print(f"  Surname: {surname}")
                print(f"  Given Name: {given_names}")
                print(f"  Document Number: {passport_number}")
                print(f"  Nationality: {nationality_code}")
                print(f"  Birth Date: {date_of_birth}")
                print(f"  Sex: {sex}")
                print(f"  Expiry Date: {expiry_date}")
                print(f"  Status: {decoded_details.get('status', 'N/A')}")
                
            else:
                print(f"⚠️ Decoded details not available or failed, using original details")
                # Fallback to original details
                document_type = details.get('document_code', 'P')
                country_code = details.get('issuer_code', '').upper()
                
                # Use manual name parsing for fallback too
                def parse_names_from_mrz_fallback(mrz_text):
                    """Parse surname and given names correctly from MRZ line 1 (fallback)"""
                    try:
                        lines = mrz_text.strip().split('\n')
                        if len(lines) < 1:
                            return "", ""
                        
                        line1 = lines[0]
                        if len(line1) < 5:
                            return "", ""
                        
                        # Extract name section (after P< + 3-char country code)
                        name_section = line1[5:]
                        
                        # Split by << to separate surname from given names
                        if '<<' in name_section:
                            parts = name_section.split('<<', 1)
                            surname_raw = parts[0]
                            given_names_raw = parts[1] if len(parts) > 1 else ""
                            
                            # Clean surname
                            surname = surname_raw.replace('<', ' ').strip()
                            
                            # Clean given names - replace < with space, remove trailing <
                            given_names = given_names_raw.rstrip('<').replace('<', ' ').strip()
                            
                            return surname, given_names
                        else:
                            # No << separator found, treat all as surname
                            surname = name_section.replace('<', ' ').strip()
                            return surname, ""
                    except Exception:
                        return "", ""
                
                # Use manual parsing for names
                surname, given_names = parse_names_from_mrz_fallback(formatted_mrz_text)
                
                # Fallback to library extraction if manual parsing fails
                if not surname:
                    surname = details.get('surname', '').replace('<', ' ').strip().upper()
                if not given_names:
                    given_names = details.get('given_name', '').replace('<', ' ').strip().upper()
                passport_number = details.get('document_number', '')
                nationality_code = details.get('nationality_code', '').upper()
                date_of_birth = details.get('birth_date', '')
                sex = details.get('sex', '<').upper()
                expiry_date = details.get('expiry_date', '')
                personal_number = details.get('optional_data', '').replace('<', '').strip()
                
                # Construct MRZ text from original details
                mrz_text = details.get('raw_text', '')
            
            # Clean country codes (remove symbols)
            country_code_clean = ''.join([c for c in country_code if c.isalpha()])[:3]
            nationality_clean = ''.join([c for c in nationality_code if c.isalpha()])[:3]
            
            print(f"🧹 CLEANED CODES:")
            print(f"  Country Code: {country_code_clean}")
            print(f"  Nationality: {nationality_clean}")
            
            # Get country details
            country_info = get_country_info(country_code_clean)
            print(f"🌍 COUNTRY INFO: {country_info}")
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
            mrz_text = formatted_mrz_text
            
            # Debug: Print and separate MRZ text into lines
            print(f"Print mrz text for debug==========================: {formatted_mrz_text}")
            print(f"===============================================88=====================================================")
            
            if mrz_text:
                # Format MRZ into proper two-line format
                formatted_mrz = format_mrz_lines(mrz_text, verbose=True)
                
                if formatted_mrz and '\n' in formatted_mrz:
                    lines = formatted_mrz.split('\n')
                    if len(lines) >= 2:
                        print(f"\n🔍 MRZ LINES SEPARATED:")
                        print(f"┌─────────────────────────────────────────────────┐")
                        print(f"│ Line 1 (44 chars): {lines[0]}                   │")
                        print(f"│ Line 2 (44 chars): {lines[1]}                   │")
                        print(f"└─────────────────────────────────────────────────┘")
                        print(f"Line 1 Length: {len(lines[0])}")
                        print(f"Line 2 Length: {len(lines[1])}")
                        
                        # Also show in simple format for easy copying
                        print(f"\n📋 COPY-READY FORMAT:")
                        print(f"{lines[0]}")
                        print(f"{lines[1]}")
                        
                        # Decode MRZ using FastMRZ _parse_mrz method (Second decode attempt)
                        print(f"\n🔍 SECOND DECODE ATTEMPT WITH FASTMRZ:")
                        try:
                            decode_fast_mrz2 = FastMRZ()
                            mrz_for_parsing2 = f"{lines[0]}\n{lines[1]}"
                            decoded_details2 = decode_fast_mrz2._parse_mrz(mrz_for_parsing2)
                            print(f"📋 SECOND DECODED DETAILS:")
                            print(decoded_details2)
                            
                            # Show validation status
                            if decoded_details2:
                                status = decoded_details2.get('status', 'UNKNOWN')
                                print(f"\n✅ VALIDATION STATUS: {status}")
                                if status != 'SUCCESS':
                                    print(f"⚠️  Status Message: {decoded_details2.get('status_message', 'No message')}")
                        except Exception as e:
                            print(f"❌ Error in second decode: {e}")
                        
                        # Use the formatted MRZ
                        mrz_text = formatted_mrz
                    else:
                        print(f"⚠ Warning: Could not split into two lines")
                else:
                    print(f"⚠ Warning: MRZ formatting failed")
            
            print(f"====================================================================================================")
            
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

            
            print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("MRZ validation checking result here.")
            mrz_for_validation =  f"{mrz_text}" 
            # print(mrz_for_validation) 
            
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
            print(f"\nField Validation Summary: {valid_count}/{total_count} fields are valid\n")
            
            # Final passport full data
            print("Final passport full data:")
            print("=" * 50)
            print(f"Document Type      : {passport_data.get('document_type', 'N/A')}")
            print(f"Country Code       : {passport_data.get('country_code', 'N/A')}")
            print(f"Country Name       : {passport_data.get('country_name', 'N/A')}")
            print(f"Surname            : {passport_data.get('surname', 'N/A')}")
            print(f"Given Names        : {passport_data.get('given_names', 'N/A')}")
            print(f"Passport Number    : {passport_data.get('passport_number', 'N/A')}")
            print(f"Nationality        : {passport_data.get('nationality', 'N/A')}")
            print(f"Date of Birth      : {passport_data.get('date_of_birth', 'N/A')}")
            print(f"Sex                : {passport_data.get('sex', 'N/A')}")
            print(f"Expiry Date        : {passport_data.get('expiry_date', 'N/A')}")
            print(f"Personal Number    : {passport_data.get('personal_number', 'N/A')}")
            print("=" * 50)
            print()
        
            
            
            if verbose:
                print(f"  ✓ Passport data extracted successfully")
                print(f"    Surname: {surname}")
                print(f"    Given Names: {given_names}")
                print(f"    Passport #: {passport_number}")
                print(f"    Country: {country_name} ({country_code_clean})")
                print(f"Test mrz test----------------------------------------------: {mrz_text}")
            
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
