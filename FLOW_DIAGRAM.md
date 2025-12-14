# Passport Scanner Flow Diagram

## NEW MULTI-LAYERED FALLBACK SYSTEM

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
│ STEP 1: PassportEye Fallback Validation                    │
│ • validate_passport_with_PassportEye_fallback()            │
│ • Image preprocessing for better OCR                        │
│ • PassportEye MRZ detection                                 │
│ • TD3 format validation & cleaning                          │
│ • Meaningful data validation (not just passport_number)     │
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
    │ ✅ Return     │         │ STEP 2: FastMRZ Fallback       │
    │ Success       │         │ • validate_passport_with_       │
    │ (Early Exit)  │         │   fastmrz_fallback()            │
    └───────────────┘         │ • FastMRZ library integration   │
                              │ • Format output to standard     │
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
                         │ ✅ Return    │        │ ❌ Return Error │
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
│ STEP 1: PassportEye Fallback Validation                    │
│ • validate_passport_with_PassportEye_fallback()            │
│ • Image preprocessing for better OCR                        │
│ • PassportEye MRZ detection                                 │
│ • TD3 format validation & cleaning                          │
│ • Meaningful data validation (not just passport_number)     │
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
    │ ✅ Return     │         │ STEP 2: FastMRZ Fallback       │
    │ Success       │         │ • validate_passport_with_       │
    │ (Early Exit)  │         │   fastmrz_fallback()            │
    └───────────────┘         │ • FastMRZ library integration   │
                              │ • Format output to standard     │
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

### STEP 1: PassportEye Fallback Validation

- **Primary Method**: PassportEye MRZ detection with preprocessing
- **TD3 Format Validation**: Ensures proper passport MRZ structure
- **TD3 Cleaning Rules**:
  - Line 1: Exactly `P<` (not `P<<`)
  - Line 2: Smart field reconstruction
  - Both lines: Exactly 44 characters
- **Meaningful Data Validation**: Checks for extractable data (not just passport_number)
- **TD3 Compliance**: Follows international standards
  - Empty passport number (filled with `<`) is valid
  - Sex field: M, F, or < (unknown)
- **Early Termination**: Returns immediately on success

### STEP 2: FastMRZ Fallback Validation

- **Secondary Method**: FastMRZ library integration
- **Standardized Output**: Formats to match system response structure
- **Error Handling**: Comprehensive exception management
- **Early Termination**: Returns immediately on success

### STEP 3: Passport Validation Checker

- **MRZ Text Validation**: Validates extracted MRZ against TD3 standards
- **Confidence Scoring**: Provides validation confidence levels
- **Threshold-Based**: Requires ≥50% confidence to proceed
- **Fallback Trigger**: Low confidence triggers AI parser (if enabled)

### STEP 4: AI Parser (Final Fallback)

- **Gemini AI Integration**: Advanced OCR and text analysis
- **Full Document Analysis**: Processes entire document, not just MRZ
- **Last Resort**: Only activated when all other methods fail
- **AI-Gated**: Only runs when `use_gemini=True`

## Key Differences Between AI=ON vs AI=OFF

| Feature                | AI=ON                                  | AI=OFF                                    |
| ---------------------- | -------------------------------------- | ----------------------------------------- |
| **Fallback Chain**     | PassportEye → FastMRZ → Validator → AI | PassportEye → FastMRZ → Validator → Error |
| **Gemini API Calls**   | ✅ Yes (Step 4 only)                   | ❌ No                                     |
| **Steps Available**    | 4 validation steps                     | 3 validation steps                        |
| **Final Fallback**     | AI Parser (Gemini)                     | Return Error                              |
| **Success Rate**       | Higher (AI rescue for difficult cases) | Lower (depends on MRZ quality)            |
| **Speed**              | Fast (early termination)               | Fast (early termination)                  |
| **Cost**               | Uses Gemini API quota (rarely)         | Free (no API calls)                       |
| **Offline Capability** | Requires internet for Step 4           | Fully offline                             |
| **Best For**           | Production, maximum accuracy           | Testing, cost-saving, offline use         |

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

1. **STEP 1: PassportEye Fallback** (`passportEye.py`)

   ```python
   def validate_passport_with_PassportEye_fallback(image, verbose=True):
       # Image preprocessing for better OCR
       # PassportEye MRZ detection
       # TD3 format validation & cleaning
       # Meaningful data validation (not just passport_number)
       # Returns: {"success": bool, "passport_data": dict, ...}
   ```

2. **STEP 2: FastMRZ Fallback** (`fastMRZ.py`)

   ```python
   def validate_passport_with_fastmrz_fallback(image, verbose=True):
       # FastMRZ library integration
       # Format output to standard response structure
       # Comprehensive error handling
       # Returns: {"success": bool, "passport_data": dict, ...}
   ```

3. **STEP 3: Passport Validation Checker** (`passport_detector.py`)

   ```python
   def passport_validation_checker(mrz_text, verbose=True):
       # MRZ text validation against TD3 standards
       # Confidence scoring (0.0 to 1.0)
       # Threshold-based validation (≥50% confidence)
       # Returns: {"is_valid": bool, "confidence_score": float, ...}
   ```

4. **STEP 4: AI Parser** (`gemini_passport_parser.py`)
   ```python
   def extract_passport_data_from_image(image_path, use_gemini=True):
       # Gemini AI OCR and text analysis
       # Full document processing
       # Advanced field extraction
       # Returns: {"success": bool, "passport_data": dict, ...}
   ```

### Multi-Layered Fallback Chain (`scanner.py` ~Line 713):

```python
def _extract_mrz(self) -> Tuple[List[str], Dict]:
    # STEP 1: PassportEye Fallback Validation
    passporteye_result = validate_passport_with_PassportEye_fallback(self.image)
    if passporteye_result.get("success", False):
        return mrz_lines, passport_data  # Early termination

    # STEP 2: FastMRZ Fallback Validation
    fastmrz_result = validate_passport_with_fastmrz_fallback(self.image)
    if fastmrz_result.get("success", False):
        return mrz_lines, passport_data  # Early termination

    # STEP 3: Passport Validation Checker
    validation_result = passport_validation_checker(mrz_text)
    if validation_result.get("is_valid", False) and confidence >= 0.5:
        # Continue to AI parser for final extraction

    # STEP 4: AI Parser (if use_gemini=True)
    if self.use_gemini:
        ai_result = extract_passport_data_from_image(image_path)
        if ai_result.get("success", False):
            return mrz_lines, passport_data

    # All methods failed
    return [], {}
```

### TD3 Validation Standards:

- **Passport Number**: Can be empty (filled with `<` characters) - valid per TD3
- **Sex Field**: Must be M, F, or < (unknown/unspecified)
- **Country Codes**: 3-letter ISO codes, symbols removed
- **Date Formats**: YYMMDD format validation
- **Line Length**: Exactly 44 characters per line
- **Meaningful Data**: At least one extractable field required

### Response Status Codes:

- **200 OK**: Valid passport data extracted successfully
- **422 Unprocessable Entity**: All validation methods failed
- **500 Internal Server Error**: System error during processing

All Gemini AI calls are properly gated behind `if self.use_gemini:` checks for offline compatibility.
