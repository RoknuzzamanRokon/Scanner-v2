"""
PassportEye MRZ extraction with preprocessing and timing
Simple implementation following the fastMRZ pattern
"""
import cv2
import time
from passporteye import read_mrz

IMAGE_PATH = None  # Will be set when calling the function

def preprocess_image(image_path, output_path="temp_preprocessed.jpg"):
    """
    Preprocess image for better MRZ detection
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize image (2x upscale)
    gray = cv2.resize(gray, (img.shape[1]*2, img.shape[0]*2))
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save preprocessed image
    cv2.imwrite(output_path, thresh)
    return output_path


def process_passport_image(image_path):
    """
    Process passport image and extract MRZ data with timing
    Similar to the fastMRZ.py functionality
    """
    global IMAGE_PATH
    IMAGE_PATH = image_path
    
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
        ocr_start = time.time()
        mrz = read_mrz(preprocessed_path)
        ocr_end = time.time()
        
        print("\n--- MRZ Parsed Fields ---")
        if mrz:
            mrz_data = mrz.to_dict()
            
            # Print all fields
            for key, value in mrz_data.items():
                print(f"{key}: {value}")
            
            # Only print selected fields
            wanted_fields = ["mrz_type", "valid_score", "raw_text",
                           "type", "country", "number", "date_of_birth",
                           "expiration_date", "nationality", "sex"]
            
            print("\n--- Selected Fields ---")
            for key in wanted_fields:
                print(f"{key}: {mrz_data.get(key, '')}")
        else:
            print("No MRZ Found.")
        
        # Validation
        val_start = time.time()
        is_valid = mrz is not None
        val_end = time.time()
        
        if is_valid:
            print("\nDocument is Valid for passport.")
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
                
                # Extract passport data from result
                passport_data = {
                    "document_type": result.get('type', 'P'),
                    "country_code": result.get('country', ''),
                    "surname": result.get('surname', ''),
                    "given_names": result.get('names', ''),
                    "passport_number": result.get('number', ''),
                    "nationality": result.get('nationality', ''),
                    "date_of_birth": result.get('date_of_birth', ''),
                    "sex": result.get('sex', ''),
                    "expiry_date": result.get('expiration_date', ''),
                    "mrz_confidence": result.get('valid_score', 0)
                }
                
                return {
                    "success": True,
                    "passport_data": passport_data,
                    "mrz_text": result.get('raw_text', ''),
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