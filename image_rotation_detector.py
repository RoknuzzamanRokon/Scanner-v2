"""
Image rotation detection using smart analysis
Automatically detects and corrects image orientation before processing
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


def detect_and_correct_rotation(image: Image.Image, verbose: bool = True) -> dict:
    """
    Detect and correct image rotation using smart analysis
    
    Args:
        image: PIL Image object
        verbose: Print debug information
        
    Returns:
        Dictionary with rotation results
    """
    
    try:
        if verbose:
            print("  → Analyzing image orientation using smart detection...")
        
        # Apply smart rotation analysis
        corrected_image, rotation_angle, confidence = smart_rotation_detection(image, verbose)
        
        rotation_needed = abs(rotation_angle) > 0.5 and confidence > 0.7
        
        result = {
            'success': True,
            'original_image': image,
            'corrected_image': corrected_image,
            'rotation_applied': rotation_angle,
            'rotation_needed': rotation_needed,
            'method_used': 'Smart Detection',
            'error': None
        }
        
        if verbose:
            if rotation_needed:
                print(f"  ✓ Image rotation corrected: {rotation_angle:.1f}° (confidence: {confidence:.2f})")
            else:
                print(f"  ✓ Image orientation is correct (confidence: {confidence:.2f})")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"  ⚠ Smart rotation detection failed: {e}")
        
        return {
            'success': False,
            'original_image': image,
            'corrected_image': image,
            'rotation_applied': 0,
            'rotation_needed': False,
            'method_used': 'Smart Detection',
            'error': str(e)
        }


def smart_rotation_detection(image: Image.Image, verbose: bool = True) -> tuple:
    """
    Smart rotation detection that distinguishes between rotated and properly oriented images
    
    Returns:
        Tuple of (corrected_image, rotation_angle, confidence)
    """
    
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    aspect_ratio = width / height
    
    if verbose:
        print(f"    → Image: {width}x{height}, aspect ratio: {aspect_ratio:.2f}")
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Line detection
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)  # Lower threshold for better detection
    
    if lines is None:
        if verbose:
            print("    → No lines detected, keeping original orientation")
        return image, 0, 0.0
    
    # Analyze line angles
    angles = []
    for i in range(min(30, len(lines))):  # Analyze more lines
        rho, theta = lines[i][0]
        angle_deg = (theta - np.pi / 2) * 180 / np.pi
        angles.append(angle_deg)
    
    # Categorize angles with tighter tolerances
    near_horizontal = [a for a in angles if abs(a) < 10]  # Within 10° of horizontal
    near_vertical = [a for a in angles if abs(abs(a) - 90) < 10]  # Within 10° of vertical
    
    if verbose:
        print(f"    → Lines: {len(lines)} total, {len(near_horizontal)} horizontal, {len(near_vertical)} vertical")
        print(f"    → Sample angles: {[f'{a:.1f}°' for a in angles[:5]]}")
    
    # Decision logic with confidence scoring
    confidence = 0.0
    rotation_angle = 0
    
    # For passport/document images, be more conservative about 90° rotations
    # Passports naturally have many vertical lines (text, borders, fields)
    
    # Strong evidence for 90° rotation - much more restrictive criteria
    if len(near_vertical) >= 10 and len(near_vertical) > len(near_horizontal) * 3 and len(near_horizontal) < 3:
        vertical_median = np.median(near_vertical)
        
        # Calculate confidence based on line count and consistency - more conservative
        line_ratio = len(near_vertical) / max(len(near_horizontal), 1)
        angle_consistency = 1.0 - (np.std(near_vertical) / 90.0) if len(near_vertical) > 1 else 1.0
        
        # Much more restrictive confidence calculation
        confidence = min(0.85, (line_ratio / 20.0) + angle_consistency * 0.4)
        
        # Additional check: ensure we have overwhelming vertical evidence
        if line_ratio > 5.0 and angle_consistency > 0.8:
            # Determine rotation direction
            if vertical_median < 0:  # Lines at ~-90°
                rotation_angle = 90  # Rotate clockwise to make them horizontal
            else:  # Lines at ~+90°
                rotation_angle = -90  # Rotate counter-clockwise
                
            if verbose:
                print(f"    → Strong vertical evidence: {len(near_vertical)} lines at {vertical_median:.1f}°")
                print(f"    → Line ratio: {line_ratio:.1f}, consistency: {angle_consistency:.2f}")
                print(f"    → Confidence: {confidence:.2f}, suggested rotation: {rotation_angle}°")
        else:
            # Not confident enough for 90° rotation
            confidence = 0.5
            rotation_angle = 0
            if verbose:
                print(f"    → Vertical lines detected but not confident enough for 90° rotation")
                print(f"    → Line ratio: {line_ratio:.1f}, consistency: {angle_consistency:.2f}")
    
    # Evidence for correct orientation - enhanced logic
    elif len(near_horizontal) >= 3:
        horizontal_median = np.median(near_horizontal)
        
        # Check if we have a good mix of horizontal and vertical lines (normal for documents)
        if len(near_vertical) > 0 and len(near_horizontal) > 0:
            h_to_v_ratio = len(near_horizontal) / len(near_vertical)
            if 0.3 <= h_to_v_ratio <= 3.0:  # Balanced mix suggests proper orientation
                confidence = 0.9
                rotation_angle = 0
                if verbose:
                    print(f"    → Balanced line distribution suggests correct orientation")
                    print(f"    → Horizontal: {len(near_horizontal)}, Vertical: {len(near_vertical)}, Ratio: {h_to_v_ratio:.2f}")
            else:
                # Small skew correction only
                if abs(horizontal_median) > 1.0 and abs(horizontal_median) < 10:
                    rotation_angle = horizontal_median
                    confidence = 0.6
                    if verbose:
                        print(f"    → Small skew correction: {rotation_angle:.1f}°")
                else:
                    rotation_angle = 0
                    confidence = 0.8
                    if verbose:
                        print(f"    → Image appears correctly oriented")
        else:
            # Only horizontal lines detected
            if abs(horizontal_median) > 1.0 and abs(horizontal_median) < 10:
                rotation_angle = horizontal_median
                confidence = 0.6
                if verbose:
                    print(f"    → Small skew correction: {rotation_angle:.1f}°")
            else:
                rotation_angle = 0
                confidence = 0.8
                if verbose:
                    print(f"    → Image appears correctly oriented")
    
    # Insufficient evidence
    else:
        confidence = 0.3
        rotation_angle = 0
        if verbose:
            print(f"    → Insufficient evidence for rotation")
    
    # Apply rotation only if very confident - higher threshold for 90° rotations
    rotation_threshold = 0.85 if abs(rotation_angle) >= 45 else 0.7
    
    if abs(rotation_angle) > 0.5 and confidence >= rotation_threshold:
        (h, w) = img_cv.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated_cv = cv2.warpAffine(img_cv, M, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
        
        # Convert back to PIL
        rotated_rgb = cv2.cvtColor(rotated_cv, cv2.COLOR_BGR2RGB)
        rotated_image = Image.fromarray(rotated_rgb)
        
        return rotated_image, rotation_angle, confidence
    else:
        return image, 0, confidence


def save_rotation_debug_images(result: dict, user_folder: Optional[str] = None) -> None:
    """
    Save debug images showing rotation detection results
    
    Args:
        result: Result dictionary from detect_and_correct_rotation
        user_folder: Optional user folder for saving debug images
    """
    original_saved = False
    corrected_saved = False
    
    try:
        from pathlib import Path
        from datetime import datetime
        
        if not result.get('success', False):
            return
        
        # Create debug folder
        if user_folder:
            debug_folder = Path(user_folder) / "rotation_debug"
        else:
            debug_folder = Path("temp") / "rotation_debug"
        
        debug_folder.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        # Always try to save original image first
        try:
            original_path = debug_folder / f"original_{timestamp}.jpg"
            
            if 'original_image' in result:
                # Convert RGBA to RGB if necessary (JPEG doesn't support transparency)
                image_to_save = result['original_image']
                if image_to_save.mode == 'RGBA':
                    # Create white background and paste RGBA image on it
                    rgb_image = Image.new('RGB', image_to_save.size, (255, 255, 255))
                    rgb_image.paste(image_to_save, mask=image_to_save.split()[-1])  # Use alpha channel as mask
                    image_to_save = rgb_image
                elif image_to_save.mode not in ['RGB', 'L']:
                    # Convert other modes to RGB
                    image_to_save = image_to_save.convert('RGB')
                
                image_to_save.save(str(original_path), 'JPEG', quality=95)
                original_saved = True
        except Exception as e:
            print(f"  ⚠ Failed to save original image: {e}")
        
        # Try to save corrected image (if different)
        try:
            if result.get('rotation_needed', False):
                angle_str = f"{result['rotation_applied']:.1f}".replace('.', '_')
                corrected_path = debug_folder / f"corrected_{angle_str}deg_{timestamp}.jpg"
                
                # Convert RGBA to RGB if necessary (JPEG doesn't support transparency)
                image_to_save = result['corrected_image']
                if image_to_save.mode == 'RGBA':
                    # Create white background and paste RGBA image on it
                    rgb_image = Image.new('RGB', image_to_save.size, (255, 255, 255))
                    rgb_image.paste(image_to_save, mask=image_to_save.split()[-1])  # Use alpha channel as mask
                    image_to_save = rgb_image
                elif image_to_save.mode not in ['RGB', 'L']:
                    # Convert other modes to RGB
                    image_to_save = image_to_save.convert('RGB')
                
                image_to_save.save(str(corrected_path), 'JPEG', quality=95)
                corrected_saved = True
        except Exception as e:
            print(f"  ⚠ Failed to save corrected image: {e}")
        
        # Final status message
        if original_saved and corrected_saved:
            print(f"  💾 Saved rotation debug images: {debug_folder}")
        elif original_saved:
            print(f"  💾 Saved original image (no rotation needed): {debug_folder}")
        else:
            print(f"  ⚠ Failed to save rotation debug images")
            
    except Exception as e:
        print(f"  ⚠ Failed to save rotation debug images: {e}")


def save_line_detection_debug(image: Image.Image, user_folder: Optional[str] = None) -> None:
    """
    Save debug image showing detected lines for rotation analysis
    
    Args:
        image: PIL Image object
        user_folder: Optional user folder for saving debug images
    """
    try:
        from pathlib import Path
        from datetime import datetime
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        # Convert to grayscale for line detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        
        # Draw lines on image
        line_img = img_cv.copy()
        if lines is not None:
            for i in range(min(10, len(lines))):
                rho, theta = lines[i][0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Create debug folder
        if user_folder:
            debug_folder = Path(user_folder) / "rotation_debug"
        else:
            debug_folder = Path("temp") / "rotation_debug"
        
        debug_folder.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        # Save edge detection result
        edges_path = debug_folder / f"edges_{timestamp}.jpg"
        cv2.imwrite(str(edges_path), edges)
        
        # Save line detection result
        lines_path = debug_folder / f"detected_lines_{timestamp}.jpg"
        cv2.imwrite(str(lines_path), line_img)
        
        print(f"  💾 Saved line detection debug images: {debug_folder}")
        
    except Exception as e:
        print(f"  ⚠ Failed to save line detection debug images: {e}")