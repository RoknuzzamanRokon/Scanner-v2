# Passport Scanner Flow Diagram

## MULTI-LAYERED FALLBACK SYSTEM (Updated)

### With AI=ON (Default)

```
┌─────────────────────────────────────────────────────────────┐
│ POST /scan?ai=on                                            │
│ use_gemini = True                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Image Loading & Preprocessing                               │
│ • Download/decode image                                     │
│ • Create user-specific temp folder                          │
│ • Convert formats (RGBA→RGB, etc.)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: FastMRZ Fallback Validation                        │
│ • validate_passport_with_fastmrz_fallback()                │
│ • FastMRZ library integration                               │
│ • TD3 format validation & cleaning                          │
│ • Format output to standard response structure              │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │Success? │
                    └────┬────┘
                         │
            ┌────────────┴────────────┐
            │ YES                     │ NO
            ▼                         ▼
    ┌───────────────┐         ┌─────────────────────────────────┐
    │ ✅ Return     │         │ STEP 2: PassportEye Fallback   │
    │ Success       │         │ • validate_passport_with_       │
    │ (Early Exit)  │         │   PassportEye_fallback()        │
    └───────────────┘         │ • Image preprocessing for OCR   │
                              │ • PassportEye MRZ detection     │
                              └────────────┬────────────────────┘
                                           │
                                      ┌────┴────┐
                                      │Success? │
                                      └────┬────┘
                                           │
                              ┌────────────┴────────────┐
                              │ YES                     │ NO
                              ▼                         ▼
                      ┌───────────────┐         ┌─────────────────────────────────┐
                      │ ✅ Return     │         │ STEP 3: Passport Validation    │
                      │ Success       │         │ Checker                         │
                      │ (Early Exit)  │         │ • passport_validation_checker() │
                      └───────────────┘         │ • MRZ text validation vs TD3    │
                                                │ • Confidence scoring            │
                                                └────────────┬────────────────────┘
                                                             │
                                                        ┌────┴────┐
                                                        │Valid &  │
                                                        │Conf≥50%?│
                                                        └────┬────┘
                                                             │
                                                ┌────────────┴────────────┐
                                                │ YES                     │ NO
                                                ▼                         ▼
                              ┌─────────────────────────────────┐      ┌──────────────────────┐
                              │ STEP 4: AI Parser (Final)       │      │Return InvalidPassport│
                              │ • gemini_passport_parser()      │      │                      │
                              │ • Gemini AI OCR & extraction    │       ──────────────────────┘
                              │ • Full text analysis            │
                              └──────────────┬──────────────────┘
                                             │
                                        ┌────┴────┐
                                        │Success? │
                                        └────┬────┘
                                             │
                                   ┌────────────┴────────────┐
                                   │ YES                     │ NO
                                   ▼                         ▼
                         ┌───────────────┐         ┌─────────────────┐
                         │ ✅ Return     │         │ ❌ Return Error │
                         │ Success       │         │ All methods     │
                         └───────────────┘         │ failed          │
                                                   └─────────────────┘
```

### With AI=OFF

```
┌─────────────────────────────────────────────────────────────┐
│ POST /scan?ai=off                                           │
│ use_gemini = False                                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Image Loading & Preprocessing                               │
│ • Download/decode image                                     │
│ • Create user-specific temp folder                          │
│ • Convert formats (RGBA→RGB, etc.)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: FastMRZ Fallback Validation                        │
│ • validate_passport_with_fastmrz_fallback()                │
│ • FastMRZ library integration                               │
│ • TD3 format validation & cleaning                          │
│ • Format output to standard response structure              │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │Success? │
                    └────┬────┘
                         │
            ┌────────────┴────────────┐
            │ YES                     │ NO
            ▼                         ▼
    ┌───────────────┐         ┌─────────────────────────────────┐
    │ ✅ Return     │         │ STEP 2: PassportEye Fallback   │
    │ Success       │         │ • validate_passport_with_       │
    │ (Early Exit)  │         │   PassportEye_fallback()        │
    └───────────────┘         │ • Image preprocessing for OCR   │
                              │ • PassportEye MRZ detection     │
                              └────────────┬────────────────────┘
                                           │
                                      ┌────┴────┐
                                      │Success? │
                                      └────┬────┘
                                           │
                              ┌────────────┴────────────┐
                              │ YES                     │ NO
                              ▼                         ▼
                      ┌───────────────┐         ┌─────────────────────────────────┐
                      │ ✅ Return     │         │ STEP 3: Passport Validation    │
                      │ Success       │         │ Checker                         │
                      │ (Early Exit)  │         │ • passport_validation_checker() │
                      └───────────────┘         │ • MRZ text validation vs TD3    │
                                                │ • Confidence scoring            │
                                                └────────────┬────────────────────┘
                                                             │
                                                        ┌────┴────┐
                                                        │Valid &  │
                                                        │Conf≥50%?│
                                                        └────┬────┘
                                                             │
                                                ┌────────────┴────────────┐
                                                │ YES                     │ NO
                                                ▼                         ▼
                                        ┌───────────────┐         ┌─────────────────┐
                                        │ ✅ Return     │         │ ❌ Return Error │
                                        │ Success       │         │ All methods     │
                                        └───────────────┘         │ failed (No AI)  │
                                                                  └─────────────────┘
```

## Multi-Layered Fallback System Features

### STEP 1: FastMRZ Fallback Validation

- **Primary Method**: FastMRZ library integration
- **Standardized Output**: Formats to match system response structure
- **TD3 Format Validation**: Ensures proper passport MRZ structure
- **MRZ Reconstruction**: Builds MRZ if raw_text unavailable
- **Error Handling**: Comprehensive exception management
- **Early Termination**: Returns immediately on success

### STEP 2: PassportEye Fallback Validation

- **Secondary Method**: PassportEye MRZ detection with preprocessing
- **Image Preprocessing**: Upscaling and thresholding for better OCR
- **TD3 Cleaning Rules**:
  - Line 1: Exactly `P<` (not `P<<`)
  - Line 2: Smart field reconstruction
  - Both lines: Exactly 44 characters
- **Meaningful Data Validation**: Checks for extractable data (not just passport_number)
- **TD3 Compliance**: Follows international standards
  - Empty passport number (filled with `<`) is valid
  - Sex field: M, F, or < (unknown)
- **Early Termination**: Returns immediately on success

### STEP 3: EasyOCR Fallback Validation

- **Tertiary Method**: EasyOCR text extraction and MRZ reconstruction
- **Pattern Detection**: Searches for MRZ patterns in extracted text
- **Data Extraction**: Extracts passport fields from full text if MRZ unavailable
- **MRZ Reconstruction**: Rebuilds MRZ from extracted passport data
- **TD3 Validation**: Internal validation via passport_validation_checker()
- **Early Termination**: Returns immediately on success

### STEP 4: Tesseract OCR Fallback Validation

- **Quaternary Method**: Tesseract OCR text extraction and MRZ reconstruction
- **Advanced Preprocessing**: Image upscaling, Gaussian blur, adaptive thresholding
- **Custom OCR Config**: Optimized character whitelist for passport documents
- **Pattern Detection**: Searches for MRZ patterns and passport keywords
- **Data Extraction**: Extracts passport fields from full text if MRZ unavailable
- **MRZ Reconstruction**: Rebuilds MRZ from extracted passport data
- **TD3 Validation**: Internal validation via passport_validation_checker()
- **Early Termination**: Returns immediately on success
- **Graceful Fallback**: Handles missing pytesseract dependency

### STEP 5: Passport Validation Checker

- **MRZ Text Validation**: Validates extracted MRZ against TD3 standards
- **Confidence Scoring**: Provides validation confidence levels
- **Threshold-Based**: Requires ≥50% confidence to proceed
- **Fallback Trigger**: Low confidence triggers AI parser (if enabled)

### STEP 6: AI Parser (Final Fallback)

- **Gemini AI Integration**: Advanced OCR and text analysis
- **Full Document Analysis**: Processes entire document, not just MRZ
- **Last Resort**: Only activated when all other methods fail
- **AI-Gated**: Only runs when `use_gemini=True`

## Key Differences Between AI=ON vs AI=OFF

| Feature                | AI=ON                                                        | AI=OFF                                                          |
| ---------------------- | ------------------------------------------------------------ | --------------------------------------------------------------- |
| **Fallback Chain**     | FastMRZ → PassportEye → EasyOCR → Tesseract → Validator → AI | FastMRZ → PassportEye → EasyOCR → Tesseract → Validator → Error |
| **Gemini API Calls**   | ✅ Yes (Step 6 only)                                         | ❌ No                                                           |
| **Steps Available**    | 6 validation steps                                           | 5 validation steps                                              |
| **Final Fallback**     | AI Parser (Gemini)                                           | Return Error                                                    |
| **Success Rate**       | Higher (AI rescue for difficult cases)                       | Lower (depends on MRZ quality)                                  |
| **Speed**              | Fast (early termination)                                     | Fast (early termination)                                        |
| **Cost**               | Uses Gemini API quota (rarely)                               | Free (no API calls)                                             |
| **Offline Capability** | Requires internet for Step 4                                 | Fully offline                                                   |
| **Best For**           | Production, maximum accuracy                                 | Testing, cost-saving, offline use                               |

## Validation Improvements (Fixed Issues)

### ✅ **TD3 Compliance Fixed**

- **Before**: Rejected documents with empty passport numbers
- **After**: Accepts empty passport numbers (filled with `<`) per TD3 standards
- **Impact**: Resolves 422 errors for valid TD3 documents

### ✅ **Meaningful Data Validation**

- **Before**: Required passport_number field specifically
- **After**: Checks for any meaningful extracted data (country, surname, dates, etc.)
- **Impact**: More flexible validation, better success rates

### ✅ **Sex Field Validation**

- **Before**: Inconsistent sex field handling
- **After**: Proper TD3 validation (M, F, or < only)
- **Impact**: Correct handling of unknown/unspecified sex

### ✅ **Early Termination**

- **Before**: All methods ran regardless of success
- **After**: Returns immediately when any method succeeds
- **Impact**: Faster response times, better performance

## Code Implementation Details

### Critical Validation Functions:

1. **STEP 1: FastMRZ Fallback** (`fastMRZ.py`)

   ```python
   def validate_passport_with_fastmrz_fallback(image, verbose=True):
       # FastMRZ library integration with temp file processing
       # MRZ reconstruction if raw_text unavailable
       # Internal TD3 validation via passport_validation_checker()
       # Format output to standard response structure
       # Returns: {"success": bool, "passport_data": dict, "mrz_text": str, ...}
   ```

2. **STEP 2: PassportEye Fallback** (`passportEye.py`)

   ```python
   def validate_passport_with_PassportEye_fallback(image, verbose=True):
       # Image preprocessing (upscaling, thresholding)
       # PassportEye MRZ detection with temp file processing
       # TD3 format validation & cleaning (P< correction, 44-char lines)
       # Internal TD3 validation via passport_validation_checker()
       # Meaningful data validation (surname, names, country, dates)
       # Returns: {"success": bool, "passport_data": dict, "mrz_text": str, ...}
   ```

3. **STEP 3: EasyOCR Fallback** (`easyOCR.py`)

   ```python
   def validate_passport_with_easyocr_fallback(image, verbose=True):
       # EasyOCR text extraction with confidence filtering
       # MRZ pattern detection and line reconstruction
       # Passport data extraction from full text if MRZ unavailable
       # MRZ reconstruction from extracted passport fields
       # Internal TD3 validation via passport_validation_checker()
       # Returns: {"success": bool, "passport_data": dict, "mrz_text": str, ...}
   ```

4. **STEP 4: Tesseract OCR Fallback** (`tesseractOCR.py`)

   ```python
   def validate_passport_with_tesseract_fallback(image, verbose=True):
       # Tesseract OCR text extraction with advanced preprocessing
       # Custom OCR configuration for passport documents
       # MRZ pattern detection and line reconstruction
       # Passport data extraction from full text if MRZ unavailable
       # MRZ reconstruction from extracted passport fields
       # Internal TD3 validation via passport_validation_checker()
       # Returns: {"success": bool, "passport_data": dict, "mrz_text": str, ...}
   ```

5. **STEP 5: Passport Validation Checker** (`passport_detector.py`)

   ```python
   def passport_validation_checker(mrz_text, verbose=True):
       # TD3 compliance validation (line count, length, format)
       # Field-by-field validation using TD3_MRZ_RULES
       # Confidence scoring (0.0 to 1.0) based on validation score
       # Threshold-based validation (≥50% confidence required)
       # Returns: {"is_valid": bool, "confidence_score": float, "reason": str, ...}
   ```

6. **STEP 6: AI Parser** (`gemini_passport_parser.py`)
   ```python
   def gemini_ocr(image_input, is_url=True, user_id=None):
       # Single-phase AI approach:
       # extract_text_with_gemini() - Full text extraction + reconstruction
       # FastMRZ validation of reconstructed MRZ
       # Multi-language pattern matching for passport fields
       # Returns: {"success": bool, "passport_data": dict, "mrz_text": str, ...}
   ```

### Multi-Layered Fallback Chain (`scanner.py` - scan_passport function):

```python
def scan_passport(image_url=None, image_base64=None, document_type="passport", use_gemini=True):
    # STEP 1: FastMRZ Fallback Validation
    fastmrz_result = validate_passport_with_fastmrz_fallback(image, verbose=True)
    if fastmrz_result.get("success", False):
        return success_response_with_timing("EasyOCR")  # Early termination

    # STEP 2: PassportEye Fallback Validation
    passporteye_result = validate_passport_with_PassportEye_fallback(image, verbose=True)
    if passporteye_result.get("success", False):
        return success_response_with_timing("PassportEye")  # Early termination

    # STEP 3: EasyOCR Fallback Validation
    easyocr_result = validate_passport_with_easyocr_fallback(image, verbose=True)
    if easyocr_result.get("success", False):
        return success_response_with_timing("EasyOCR")  # Early termination

    # STEP 4: Tesseract OCR Fallback Validation
    tesseract_result = validate_passport_with_tesseract_fallback(image, verbose=True)
    if tesseract_result.get("success", False):
        return success_response_with_timing("Tesseract")  # Early termination

    # STEP 5: Passport Validation Checker
    mrz_text = fastmrz_result.get("mrz_text", "") or passporteye_result.get("mrz_text", "") or easyocr_result.get("mrz_text", "") or tesseract_result.get("mrz_text", "")
    if mrz_text:
        validation_result = passport_validation_checker(mrz_text, verbose=True)
        if validation_result.get("is_valid", False) and confidence >= 0.5:
            if not use_gemini:  # AI=OFF mode
                return success_response_with_timing("Tesseract")  # Early termination
            # AI=ON: Continue to Step 6 for enhanced extraction

    # STEP 6: AI Parser (if use_gemini=True)
    if use_gemini:
        ai_result = gemini_ocr(image_input, is_url=bool(image_url), user_id=user_id)
        if ai_result.get("success", False):
            return success_response_with_timing("AI")

    # All methods failed
    return failure_response_with_errors()
```

### TD3 Validation Standards (td3_validation_check.py):

- **Line Format**: Exactly 2 lines, 44 characters each
- **Line 1**: `P<CCC` + Name field (surname<<given_names) padded with `<`
- **Line 2**: Passport# + Check + Country + DOB + Check + Sex + Expiry + Check + Personal# + Check + Final
- **Document Type**: Must match `P[<A-Z]` pattern (usually `P<`)
- **Country Codes**: 3-letter ISO codes (A-Z), symbols cleaned
- **Sex Field**: Must be M, F, X, or < (unknown/unspecified)
- **Date Formats**: YYMMDD format (6 digits) for birth_date and expiry_date
- **Passport Number**: Can be empty (filled with `<`) - valid per TD3 standards
- **Confidence Scoring**: Field-by-field validation, ≥50% required to pass
- **Meaningful Data**: Surname, given names, country, or dates must be extractable

### Response Status Codes:

- **200 OK**: Valid passport data extracted successfully
- **422 Unprocessable Entity**: All validation methods failed
- **500 Internal Server Error**: System error during processing

## Implementation Notes

### Error Handling & Robustness:

- **Comprehensive Exception Handling**: Each step wrapped in try-catch blocks
- **Temp File Management**: Automatic cleanup of temporary files in all methods
- **Image Format Conversion**: RGBA→RGB conversion for compatibility
- **Preprocessing**: Image upscaling and thresholding for better OCR
- **Validation Integration**: Each extraction method validates via TD3 checker

### AI Integration:

- **Gated AI Calls**: All Gemini calls properly gated behind `if use_gemini:` checks
- **Single-Phase AI Approach**: Full text extraction and data reconstruction
- **Multi-Language Support**: Pattern matching for multiple languages in AI parser
- **Offline Compatibility**: System works fully offline when AI=OFF

### Performance Optimizations:

- **Early Termination**: Returns immediately when any method succeeds
- **Step Timing**: Detailed timing information for each validation step
- **User-Specific Temp Folders**: Isolated processing for concurrent users
- **Minimal API Usage**: AI only called as final fallback, reducing costs
