import json
from scanner import scan_passport

# Example passport image URL (replace with a real image URL or base64 string)
sample_url = "https://example.com/sample-passport.jpg"

result = scan_passport(image_url=sample_url, use_gemini=True)
print(json.dumps(result, indent=2))
