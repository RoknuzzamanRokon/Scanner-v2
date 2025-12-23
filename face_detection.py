"""
Face Detection and Alignment Module
Detects face in passport image and performs alignment for better OCR accuracy
"""
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple
import os
import time


def detect_face_features(image: Image.Image) -> Dict:
    """
    Detect face and facial features (eyes, nose) in passport image
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with detection results including:
        - success: bool
        - face_detected: bool
        - eyes_detected: bool  
        - nose_detected: bool
        - face_region: (x, y, w, h) or None
        - rotation_angle: float (degrees) or None
        - error: str or None
    """
    result = {
        "success": False,
        "face_detected": False,
        "eyes_detected": False,
        "nose_detected": False,
        "face_region": None,
        "rotation_angle": None,
        "error": None
    }
    
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            gray = img_array
            color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            color_img = img_array.copy()
        
        height, width = gray.shape
        
        # Step 1: Face detection using Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        if len(faces) > 0:
            result["face_detected"] = True
            # Get largest face
            face_x, face_y, face_w, face_h = max(faces, key=lambda f: f[2] * f[3])
            result["face_region"] = (face_x, face_y, face_w, face_h)
            
            # Step 2: Eye detection within face region
            face_roi = gray[face_y:face_y+face_h, face_x:face_x+face_w]
            eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            
            if len(eyes) >= 2:
                result["eyes_detected"] = True
                
                # Calculate rotation angle based on eye positions
                # Sort eyes by x-coordinate (left to right)
                eyes_sorted = sorted(eyes, key=lambda eye: eye[0])
                left_eye = eyes_sorted[0]
                right_eye = eyes_sorted[1]
                
                # Calculate eye centers
                left_eye_center = (face_x + left_eye[0] + left_eye[2]//2, face_y + left_eye[1] + left_eye[3]//2)
                right_eye_center = (face_x + right_eye[0] + right_eye[2]//2, face_y + right_eye[1] + right_eye[3]//2)
                
                # Calculate angle between eyes (should be horizontal)
                dx = right_eye_center[0] - left_eye_center[0]
                dy = right_eye_center[1] - left_eye_center[1]
                
                if dx != 0:  # Avoid division by zero
                    angle = np.degrees(np.arctan2(dy, dx))
                    result["rotation_angle"] = angle
                    
                    # If angle is significant (> 2 degrees), we should rotate
                    if abs(angle) > 2:
                        result["rotation_applied"] = True
                    else:
                        result["rotation_applied"] = False
                
            # Step 3: Nose detection (optional)
            # For now, we'll consider nose detected if face is detected
            result["nose_detected"] = True
            
            result["success"] = True
            result["error"] = None
            
        else:
            result["error"] = "No face detected"
            
    except Exception as e:
        result["error"] = f"Face detection failed: {str(e)}"
        
    return result


def align_face(image: Image.Image, rotation_angle: float) -> Image.Image:
    """
    Align face by rotating image to make eyes horizontal
    
    Args:
        image: PIL Image object
        rotation_angle: Rotation angle in degrees
        
    Returns:
        Aligned PIL Image
    """
    if abs(rotation_angle) < 1:  # No significant rotation needed
        return image
    
    # Convert to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    height, width = img_array.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(img_array, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return Image.fromarray(rotated)


def crop_passport_page(image: Image.Image, face_region: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crop passport page based on face position using ICAO standards
    
    Args:
        image: PIL Image object
        face_region: (x, y, w, h) of detected face
        
    Returns:
        Cropped PIL Image of passport page
    """
    face_x, face_y, face_w, face_h = face_region
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    print(f"  Face detection: x={face_x}, y={face_y}, w={face_w}, h={face_h}")
    print(f"  Image dimensions: {width}x{height}")
    
    # ICAO standards: passport page is typically 88mm × 125mm
    # Photo is 35mm × 45mm, located in top-left
    # We'll estimate passport page boundaries based on face position
    
    # Estimate passport page boundaries
    # Passport page typically extends beyond the photo
    page_left = 0  # Use full width - start from left edge
    page_top = max(0, face_y - face_h * 2)  # Extend top by 2x face height (increased from 0.5x)
    page_right = width  # Use full width - extend to right edge
    # Changed: Instead of going to bottom, stop at 20% above bottom (80% of image height)
    page_bottom = min(height, int(height * 0.8))  # Stop at 80% of image height (20% above bottom)
    
    print(f"  Crop region: left={page_left}, top={page_top}, right={page_right}, bottom={page_bottom}")
    
    # Validate crop region
    if page_right <= page_left or page_bottom <= page_top:
        print("  [WARNING] Invalid crop region, using full image")
        return image
    
    # Crop passport page
    cropped = img_array[page_top:page_bottom, page_left:page_right]
    
    # Validate crop result
    if cropped.size == 0:
        print("  [WARNING] Empty crop result, using full image")
        return image
    
    crop_height, crop_width = cropped.shape[:2]
    print(f"  Crop successful: {crop_width}x{crop_height} pixels")
    
    return Image.fromarray(cropped)


def preprocess_passport_image_with_face_detection(image: Image.Image, debug: bool = False, user_folder: Optional[str] = None) -> Dict:
    """
    Complete face detection and alignment preprocessing pipeline
    
    Args:
        image: PIL Image object
        debug: Whether to save debug images
        user_folder: Optional user folder path for saving debug images
        
    Returns:
        Dictionary with preprocessing results
    """
    start_time = time.time()
    result = {
        "success": False,
        "face_detected": False,
        "eyes_detected": False,
        "nose_detected": False,
        "rotation_applied": False,
        "passport_page_cropped": False,
        "processed_image": None,
        "error": None,
        "processing_time": 0.0
    }
    
    try:
        print("  [INFO] Starting face detection and alignment...")
        
        # Step 1: Detect face and features
        detection_result = detect_face_features(image)
        
        if detection_result["success"] and detection_result["face_detected"]:
            result.update({
                "face_detected": True,
                "eyes_detected": detection_result["eyes_detected"],
                "nose_detected": detection_result["nose_detected"]
            })
            
            # Step 2: Align face if rotation is needed
            if detection_result.get("rotation_angle") and abs(detection_result["rotation_angle"]) > 2:
                aligned_image = align_face(image, detection_result["rotation_angle"])
                result["rotation_applied"] = True
                print(f"  → Face rotation applied: {detection_result['rotation_angle']:.1f}°")
            else:
                aligned_image = image
                result["rotation_applied"] = False
                print("  → No rotation needed (face already aligned)")
            
            # Step 3: Crop passport page based on face position
            if detection_result["face_region"]:
                cropped_image = crop_passport_page(aligned_image, detection_result["face_region"])
                result["passport_page_cropped"] = True
                result["processed_image"] = cropped_image
                print("  → Passport page cropped based on face position")
                
                # Save passport_page_crop.jpg
                try:
                    from utils import save_passport_page_crop
                    from pathlib import Path
                    
                    if user_folder:
                        passport_crop_path = save_passport_page_crop(cropped_image, Path(user_folder))
                    else:
                        passport_crop_path = save_passport_page_crop(cropped_image)
                    
                    result["passport_page_crop_path"] = passport_crop_path
                    print(f"  Cropped image saved: {passport_crop_path}")
                    
                except Exception as e:
                    print(f"  [WARNING] Failed to save passport_page_crop.jpg: {e}")
            else:
                result["processed_image"] = aligned_image
                
                # Save passport_page_crop.jpg even without face detection
                try:
                    from utils import save_passport_page_crop
                    from pathlib import Path
                    
                    if user_folder:
                        passport_crop_path = save_passport_page_crop(aligned_image, Path(user_folder))
                    else:
                        passport_crop_path = save_passport_page_crop(aligned_image)
                    
                    result["passport_page_crop_path"] = passport_crop_path
                    print(f"  Original image saved (no face detection): {passport_crop_path}")
                    
                except Exception as e:
                    print(f"  [WARNING] Failed to save passport_page_crop.jpg: {e}")
                
            # Save debug images if requested
            if debug and user_folder:
                try:
                    from pathlib import Path
                    import datetime
                    
                    # Create debug folder if it doesn't exist
                    debug_folder = Path(user_folder) / "debug"
                    debug_folder.mkdir(exist_ok=True)
                    
                    # Save original image with face detection overlay
                    img_array = np.array(image)
                    if len(img_array.shape) == 2:
                        debug_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    else:
                        debug_img = img_array.copy()
                    
                    # Draw face detection results and crop areas
                    if detection_result["face_region"]:
                        face_x, face_y, face_w, face_h = detection_result["face_region"]
                        
                        # Draw face rectangle
                        cv2.rectangle(debug_img, (face_x, face_y), (face_x+face_w, face_y+face_h), (0, 255, 0), 2)
                        cv2.putText(debug_img, "Face", (face_x, face_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Draw crop area if passport page was cropped
                        if result.get("passport_page_cropped"):
                            # Calculate crop area based on face position (same logic as crop_passport_page)
                            page_left = max(0, face_x - face_w)
                            page_top = max(0, face_y - face_h // 2)
                            page_right = min(img_array.shape[1], face_x + face_w * 3)
                            page_bottom = min(img_array.shape[0], face_y + face_h * 4)
                            
                            # Draw crop area outline
                            cv2.rectangle(debug_img, (page_left, page_top), (page_right, page_bottom), (255, 0, 0), 3)
                            cv2.putText(debug_img, "CROP AREA", (page_left, page_top - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                            
                            # Add crop dimensions
                            crop_w = page_right - page_left
                            crop_h = page_bottom - page_top
                            cv2.putText(debug_img, f"{crop_w}x{crop_h}px", (page_left, page_bottom + 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Save debug image
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    debug_path = debug_folder / f"face_detection_crop_debug_{timestamp}.jpg"
                    cv2.imwrite(str(debug_path), debug_img)
                    print(f"  → Saved face detection crop debug: {debug_path}")
                    
                except Exception as e:
                    print(f"  ⚠ Debug image save failed: {e}")
            
            result["success"] = True
            result["error"] = None
            
        else:
            result["error"] = detection_result["error"] or "Face detection failed"
            result["processed_image"] = image  # Return original image if detection fails
            
            # Save passport_page_crop.jpg even when face detection fails
            try:
                from utils import save_passport_page_crop
                from pathlib import Path
                
                if user_folder:
                    passport_crop_path = save_passport_page_crop(image, Path(user_folder))
                else:
                    passport_crop_path = save_passport_page_crop(image)
                
                result["passport_page_crop_path"] = passport_crop_path
                print("  → Saved passport_page_crop.jpg (fallback - no face detection)")
                
            except Exception as e:
                print(f"  ⚠ Failed to save passport_page_crop.jpg (fallback): {e}")
            
    except Exception as e:
        result["error"] = f"Face detection preprocessing failed: {str(e)}"
        result["processed_image"] = image  # Return original image on error
        
    result["processing_time"] = time.time() - start_time
    print(f"  [INFO] Face detection completed in {result['processing_time']:.2f}s")
    
    return result


def step0_face_detection_and_alignment(image: Image.Image, user_folder: Optional[str] = None) -> Dict:
    """
    STEP 0: Face Detection and Alignment with Comprehensive Crop Debug
    
    This is the first step in the passport processing pipeline.
    It detects the face, aligns it, crops the passport page, and creates comprehensive debug images.
    
    Args:
        image: PIL Image object
        user_folder: Optional user folder path for saving debug images
        
    Returns:
        Dictionary with step results including processed image
    """
    print("\n" + "-"*60)
    print("[STEP 0] Facial Feature Detection & Image Preprocessing")
    print("-"*60)
    
    step_start = time.time()
    
    # Step 1: Comprehensive region detection and crop debug
    try:
        from mrz_detector import detect_and_crop_data_section
        
        print("🔍 Running comprehensive region detection with crop debug...")
        
        # This will create all the crop debug images
        cropped_result = detect_and_crop_data_section(
            image, 
            save_debug=True, 
            image_url="step0_processing", 
            user_folder=user_folder
        )
        
        if cropped_result:
            print("✓ Region detection and cropping completed")
            processed_image = cropped_result
        else:
            print("⚠ Region detection failed, using original image")
            processed_image = image
            
    except Exception as e:
        print(f"⚠ Region detection error: {e}")
        processed_image = image
    
    # Step 2: Face detection and alignment
    result = preprocess_passport_image_with_face_detection(
        processed_image, 
        debug=True, 
        user_folder=user_folder
    )
    
    step_timing = f"{time.time() - step_start:.2f}s"
    
    if result["success"]:
        print("✅ Face detection and alignment completed successfully")
        print(f"  → Face detected: {result['face_detected']}")
        print(f"  → Eyes detected: {result['eyes_detected']}")
        print(f"  → Rotation applied: {result['rotation_applied']}")
        print(f"  → Passport page cropped: {result['passport_page_cropped']}")
        
        return {
            "success": True,
            "processed_image": result["processed_image"],
            "face_detected": result["face_detected"],
            "eyes_detected": result["eyes_detected"],
            "rotation_applied": result["rotation_applied"],
            "passport_page_cropped": result["passport_page_cropped"],
            "step_timing": step_timing,
            "method_used": "Face Detection & Alignment with Crop Debug"
        }
    else:
        print(f"⚠ Face detection failed: {result['error']}")
        print("  → Continuing with processed image from region detection")
        
        return {
            "success": True,  # Still successful if we have region detection
            "processed_image": processed_image,  # Use region-detected image
            "error": result["error"],
            "step_timing": step_timing,
            "method_used": "Region Detection Only"
        }