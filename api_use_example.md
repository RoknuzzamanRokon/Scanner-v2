# Document Scanner API - Usage Examples

This document provides comprehensive examples of how to use the Document Scanner API endpoints.

## Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints Overview](#endpoints-overview)
- [Document Scanning](#document-scanning)
- [Step Configuration Management](#step-configuration-management)
- [Error Handling](#error-handling)
- [Code Examples](#code-examples)

## Base URL

```
Local Development: http://localhost:8000
Production: https://your-domain.com
```

## Authentication

Currently, no authentication is required. All endpoints are publicly accessible.

## Endpoints Overview

| Method | Endpoint       | Description                    |
| ------ | -------------- | ------------------------------ |
| GET    | `/`            | Serves the web UI              |
| GET    | `/health`      | Health check                   |
| POST   | `/scan`        | Scan document and extract data |
| GET    | `/step-config` | Get current step configuration |
| POST   | `/step-config` | Update step configuration      |

---

## Document Scanning

### POST `/scan`

Extract passport/ID card data from images or PDF files.

#### Request Parameters

| Parameter             | Type   | Required    | Description                                    |
| --------------------- | ------ | ----------- | ---------------------------------------------- |
| `image_type`          | string | Yes         | `"file"` for URL or `"base64"` for base64 data |
| `documents_image_url` | string | Conditional | Required if `image_type="file"`                |
| `image_base64`        | string | Conditional | Required if `image_type="base64"`              |
| `documents_type`      | string | No          | Document type (default: `"passport"`)          |
| `step_config`         | object | No          | Override step configuration for this scan      |

#### Query Parameters

| Parameter | Type   | Default | Description                 |
| --------- | ------ | ------- | --------------------------- |
| `ai`      | string | `"on"`  | AI usage: `"on"` or `"off"` |

### Example 1: Scan Image from URL

```bash
curl -X POST "http://localhost:8000/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "image_type": "file",
    "documents_image_url": "https://example.com/passport.jpg",
    "documents_type": "passport"
  }'
```

### Example 2: Scan Base64 Image

```bash
curl -X POST "http://localhost:8000/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "image_type": "base64",
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "documents_type": "passport"
  }'
```

### Example 3: Scan with AI Disabled

```bash
curl -X POST "http://localhost:8000/scan?ai=off" \
  -H "Content-Type: application/json" \
  -d '{
    "image_type": "file",
    "documents_image_url": "https://example.com/passport.jpg",
    "documents_type": "passport"
  }'
```

### Example 4: Scan with Custom Step Configuration

```bash
curl -X POST "http://localhost:8000/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "image_type": "file",
    "documents_image_url": "https://example.com/passport.jpg",
    "documents_type": "passport",
    "step_config": {
      "STEP1": false,
      "STEP2": true,
      "STEP3": true,
      "STEP4": false,
      "STEP5": true,
      "STEP6": false
    }
  }'
```

### Example 5: Scan PDF Document

```bash
curl -X POST "http://localhost:8000/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "image_type": "base64",
    "image_base64": "JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwo...",
    "documents_type": "passport"
  }'
```

### Success Response

```json
{
  "success": true,
  "passport_data": {
    "processVia": "FastMRZ",
    "document_type": "P",
    "country_code": "USA",
    "surname": "SMITH",
    "given_names": "JOHN DAVID",
    "passport_number": "123456789",
    "nationality": "American",
    "date_of_birth": "1990-01-15",
    "sex": "M",
    "expiry_date": "2030-01-15",
    "personal_number": "1234567890123",
    "country_name": "United States",
    "country_iso": "US"
  },
  "mrz_text": "P<USASMITH<<JOHN<DAVID<<<<<<<<<<<<<<<<<<<<<<\n1234567890USA9001151M30011511234567890123<45",
  "working_process_step": {
    "step1_fastmrz": "FastMRZ",
    "step2_passporteye": "Skipped (Early Exit)",
    "step3_easyocr": "Skipped (Early Exit)",
    "step4_tesseract": "Skipped (Early Exit)",
    "step5_validation": "Skipped (Early Exit)",
    "step6_ai_parser": "Skipped (Early Exit)"
  },
  "step_timings": {
    "image_loading": "0.45s",
    "step1_fastmrz": "1.23s",
    "step2_passporteye": "0.00s",
    "step3_easyocr": "0.00s",
    "step4_tesseract": "0.00s",
    "step5_validation": "0.00s"
  },
  "total_time": "1.68s",
  "error": "",
  "validation_reason": "",
  "validation_confidence": ""
}
```

### Error Response

```json
{
  "success": false,
  "passport_data": {},
  "mrz_text": "",
  "working_process_step": {
    "step1_fastmrz": "Failed",
    "step2_passporteye": "Failed",
    "step3_easyocr": "Failed",
    "step4_tesseract": "Failed",
    "step5_validation": "Skipped (No MRZ)",
    "step6_ai_parser": "Failed"
  },
  "step_timings": {
    "image_loading": "0.30s",
    "step1_fastmrz": "1.20s",
    "step2_passporteye": "2.15s",
    "step3_easyocr": "8.45s",
    "step4_tesseract": "3.20s",
    "step5_validation": "0.00s"
  },
  "total_time": "15.30s",
  "error": "FastMRZ: No MRZ detected | PassportEye: No MRZ detected | EasyOCR: No MRZ detected | Tesseract: No MRZ detected | AI: Processing failed",
  "validation_reason": "All extraction methods failed",
  "validation_confidence": "0%"
}
```

---

## Step Configuration Management

### GET `/step-config`

Get the current step configuration from the backend.

#### Example Request

```bash
curl -X GET "http://localhost:8000/step-config"
```

#### Response

```json
{
  "success": true,
  "steps": {
    "STEP1": true,
    "STEP2": true,
    "STEP3": true,
    "STEP4": true,
    "STEP5": false,
    "STEP6": false
  },
  "message": "Step configuration retrieved successfully"
}
```

### POST `/step-config`

Update the step configuration in the backend.

#### Request Body

```json
{
  "steps": {
    "STEP1": true,
    "STEP2": true,
    "STEP3": false,
    "STEP4": false,
    "STEP5": true,
    "STEP6": true
  }
}
```

#### Example Request

```bash
curl -X POST "http://localhost:8000/step-config" \
  -H "Content-Type: application/json" \
  -d '{
    "steps": {
      "STEP1": true,
      "STEP2": true,
      "STEP3": false,
      "STEP4": false,
      "STEP5": true,
      "STEP6": true
    }
  }'
```

#### Response

```json
{
  "success": true,
  "steps": {
    "STEP1": true,
    "STEP2": true,
    "STEP3": false,
    "STEP4": false,
    "STEP5": true,
    "STEP6": true
  },
  "message": "Step configuration updated successfully"
}
```

### Step Descriptions

| Step  | Name        | Description                       |
| ----- | ----------- | --------------------------------- |
| STEP1 | FastMRZ     | FastMRZ fallback validation       |
| STEP2 | PassportEye | PassportEye fallback validation   |
| STEP3 | EasyOCR     | EasyOCR fallback validation       |
| STEP4 | Tesseract   | Tesseract OCR fallback            |
| STEP5 | Validation  | Passport validation checker       |
| STEP6 | AI Parser   | Gemini AI parser (final fallback) |

---

## Error Handling

### HTTP Status Codes

| Code | Description                                                 |
| ---- | ----------------------------------------------------------- |
| 200  | Success                                                     |
| 422  | Unprocessable Entity (validation failed, extraction failed) |
| 500  | Internal Server Error                                       |

### Common Error Responses

#### Invalid Request Format

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "image_type"],
      "msg": "Field required"
    }
  ]
}
```

#### Image Download Failed

```json
{
  "success": false,
  "error": "System error: Failed to download image: 404 Client Error: Not Found for url: https://example.com/nonexistent.jpg",
  "passport_data": {},
  "mrz_text": "",
  "working_process_step": {},
  "step_timings": {},
  "total_time": "1.25s"
}
```

#### All Extraction Methods Failed

```json
{
  "success": false,
  "error": "FastMRZ: No MRZ detected | PassportEye: No MRZ detected | EasyOCR: No MRZ detected | Tesseract: No MRZ detected | AI: Processing failed",
  "validation_reason": "All extraction methods failed",
  "validation_confidence": "0%"
}
```

---

## Code Examples

### Python Example

```python
import requests
import base64

# Example 1: Scan from URL
def scan_from_url(image_url):
    url = "http://localhost:8000/scan"
    payload = {
        "image_type": "file",
        "documents_image_url": image_url,
        "documents_type": "passport"
    }

    response = requests.post(url, json=payload)
    return response.json()

# Example 2: Scan from local file
def scan_from_file(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    url = "http://localhost:8000/scan"
    payload = {
        "image_type": "base64",
        "image_base64": encoded_string,
        "documents_type": "passport"
    }

    response = requests.post(url, json=payload)
    return response.json()

# Example 3: Scan with custom step configuration
def scan_with_custom_steps(image_url, steps):
    url = "http://localhost:8000/scan"
    payload = {
        "image_type": "file",
        "documents_image_url": image_url,
        "documents_type": "passport",
        "step_config": steps
    }

    response = requests.post(url, json=payload)
    return response.json()

# Example 4: Get step configuration
def get_step_config():
    url = "http://localhost:8000/step-config"
    response = requests.get(url)
    return response.json()

# Example 5: Update step configuration
def update_step_config(steps):
    url = "http://localhost:8000/step-config"
    payload = {"steps": steps}

    response = requests.post(url, json=payload)
    return response.json()

# Usage examples
if __name__ == "__main__":
    # Scan passport from URL
    result = scan_from_url("https://example.com/passport.jpg")
    print("Scan result:", result)

    # Scan with only FastMRZ and AI Parser
    custom_steps = {
        "STEP1": True,   # FastMRZ
        "STEP2": False,  # PassportEye
        "STEP3": False,  # EasyOCR
        "STEP4": False,  # Tesseract
        "STEP5": False,  # Validation
        "STEP6": True    # AI Parser
    }
    result = scan_with_custom_steps("https://example.com/passport.jpg", custom_steps)
    print("Custom scan result:", result)

    # Get current configuration
    config = get_step_config()
    print("Current config:", config)
```

### JavaScript Example

```javascript
// Example 1: Scan from URL
async function scanFromUrl(imageUrl) {
  const response = await fetch("http://localhost:8000/scan", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      image_type: "file",
      documents_image_url: imageUrl,
      documents_type: "passport",
    }),
  });

  return await response.json();
}

// Example 2: Scan from file input
async function scanFromFile(fileInput) {
  const file = fileInput.files[0];
  const base64 = await fileToBase64(file);

  const response = await fetch("http://localhost:8000/scan", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      image_type: "base64",
      image_base64: base64,
      documents_type: "passport",
    }),
  });

  return await response.json();
}

// Helper function to convert file to base64
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = (error) => reject(error);
  });
}

// Example 3: Get step configuration
async function getStepConfig() {
  const response = await fetch("http://localhost:8000/step-config");
  return await response.json();
}

// Example 4: Update step configuration
async function updateStepConfig(steps) {
  const response = await fetch("http://localhost:8000/step-config", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ steps }),
  });

  return await response.json();
}

// Usage examples
async function examples() {
  try {
    // Scan passport
    const result = await scanFromUrl("https://example.com/passport.jpg");
    console.log("Scan result:", result);

    // Get current configuration
    const config = await getStepConfig();
    console.log("Current config:", config);

    // Update configuration
    const newConfig = {
      STEP1: true,
      STEP2: false,
      STEP3: true,
      STEP4: false,
      STEP5: true,
      STEP6: false,
    };
    const updateResult = await updateStepConfig(newConfig);
    console.log("Update result:", updateResult);
  } catch (error) {
    console.error("Error:", error);
  }
}
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Scan passport from URL
curl -X POST "http://localhost:8000/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "image_type": "file",
    "documents_image_url": "https://example.com/passport.jpg",
    "documents_type": "passport"
  }'

# Scan with AI disabled
curl -X POST "http://localhost:8000/scan?ai=off" \
  -H "Content-Type: application/json" \
  -d '{
    "image_type": "file",
    "documents_image_url": "https://example.com/passport.jpg",
    "documents_type": "passport"
  }'

# Get step configuration
curl -X GET "http://localhost:8000/step-config"

# Update step configuration
curl -X POST "http://localhost:8000/step-config" \
  -H "Content-Type: application/json" \
  -d '{
    "steps": {
      "STEP1": true,
      "STEP2": true,
      "STEP3": false,
      "STEP4": false,
      "STEP5": true,
      "STEP6": false
    }
  }'

# Scan with custom step configuration
curl -X POST "http://localhost:8000/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "image_type": "file",
    "documents_image_url": "https://example.com/passport.jpg",
    "documents_type": "passport",
    "step_config": {
      "STEP1": false,
      "STEP2": false,
      "STEP3": true,
      "STEP4": false,
      "STEP5": false,
      "STEP6": false
    }
  }'
```

---

## Processing Steps Explained

### Step Configuration Rules

1. **Backend Authority**: Steps disabled in `step_config.txt` cannot be enabled via API
2. **Frontend Override**: Steps enabled in backend can be disabled for specific scans
3. **Fallback System**: Steps run in sequence until one succeeds (early exit)

### Step Flow

```
STEP1 (FastMRZ) → Success? Return result
    ↓ Failed
STEP2 (PassportEye) → Success? Return result
    ↓ Failed
STEP3 (EasyOCR) → Success? Return result
    ↓ Failed
STEP4 (Tesseract) → Success? Return result
    ↓ Failed
STEP5 (Validation) → Validate extracted data
    ↓
STEP6 (AI Parser) → Final fallback with AI
```

### Performance Tips

1. **Use AI mode** (`ai=on`) for better accuracy with complex documents
2. **Disable unnecessary steps** to improve processing speed
3. **Use appropriate image quality** (300+ DPI recommended)
4. **Ensure good lighting** and minimal skew in images

---

## Support

For issues or questions:

- Check the logs in the server console
- Verify image URL accessibility
- Ensure proper image format (JPG, PNG, PDF)
- Check step configuration compatibility

## Version

API Version: 2.1
Last Updated: December 2024
