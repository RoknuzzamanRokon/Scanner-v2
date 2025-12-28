"""
Debug script for testing Hough line rotation detection on a specific image
"""
import cv2
import numpy as np
from PIL import Image
from image_rotation_detector import detect_and_correct_rotation, save_line_detection_debug
import os

def debug_rotation_detection(image_path):
    """
    Debug rotation detection with detailed analysis
    """
    print(f"🔍 Debugging rotation detection for: {image_path}")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    try:
        # Load image
        image = Image.open(image_path)
        print(f"✅ Image loaded successfully: {image.size} {image.mode}")
        
        # Run rotation detection with verbose output
        print("\n" + "-"*50)
        print("ROTATION DETECTION ANALYSIS")
        print("-"*50)
        
        result = detect_and_correct_rotation(image, verbose=True)
        
        print("\n" + "-"*50)
        print("DETAILED RESULTS")
        print("-"*50)
        print(f"Success: {result['success']}")
        print(f"Rotation needed: {result['rotation_needed']}")
        print(f"Rotation applied: {result['rotation_applied']:.2f}°")
        print(f"Method used: {result['method_used']}")
        if result['error']:
            print(f"Error: {result['error']}")
        
        # Save debug images
        debug_folder = "debug_output"
        os.makedirs(debug_folder, exist_ok=True)
        
        # Save original
        original_debug_path = os.path.join(debug_folder, "debug_original.jpg")
        image.save(original_debug_path)
        print(f"\n💾 Saved original for debug: {original_debug_path}")
        
        # Save corrected (if rotated)
        if result['rotation_needed']:
            corrected_debug_path = os.path.join(debug_folder, f"debug_corrected_{result['rotation_applied']:.1f}deg.jpg")
            result['corrected_image'].save(corrected_debug_path)
            print(f"💾 Saved corrected image: {corrected_debug_path}")
        
        # Save line detection debug
        save_line_detection_debug(image, debug_folder)
        
        # Additional detailed analysis
        print("\n" + "-"*50)
        print("DETAILED LINE ANALYSIS")
        print("-"*50)
        
        detailed_line_analysis(image)
        
        return result
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()
        return None

def detailed_line_analysis(image):
    """
    Perform detailed line analysis with multiple parameter sets
    """
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        print(f"Image dimensions: {gray.shape}")
        
        # Test different Canny parameters
        canny_params = [
            (50, 150),
            (30, 100),
            (100, 200),
            (20, 80)
        ]
        
        for i, (low, high) in enumerate(canny_params):
            print(f"\nCanny parameters {i+1}: low={low}, high={high}")
            edges = cv2.Canny(gray, low, high, apertureSize=3)
            
            # Count edge pixels
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_percentage = (edge_pixels / total_pixels) * 100
            print(f"  Edge pixels: {edge_pixels} ({edge_percentage:.2f}% of image)")
            
            # Test different Hough parameters
            hough_params = [
                (200, "standard"),
                (150, "relaxed"),
                (100, "very_relaxed"),
                (300, "strict")
            ]
            
            for threshold, desc in hough_params:
                lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
                
                if lines is not None:
                    num_lines = len(lines)
                    print(f"    Hough threshold {threshold} ({desc}): {num_lines} lines detected")
                    
                    if num_lines > 0:
                        # Analyze angles
                        angles = []
                        for j in range(min(10, num_lines)):
                            rho, theta = lines[j][0]
                            angle_deg = (theta - np.pi / 2) * 180 / np.pi
                            angles.append(angle_deg)
                        
                        if angles:
                            median_angle = np.median(angles)
                            mean_angle = np.mean(angles)
                            std_angle = np.std(angles)
                            
                            print(f"      Angles (first 10): {[f'{a:.1f}' for a in angles[:5]]}")
                            print(f"      Median: {median_angle:.2f}°, Mean: {mean_angle:.2f}°, Std: {std_angle:.2f}°")
                else:
                    print(f"    Hough threshold {threshold} ({desc}): No lines detected")
            
            # Save edge detection result for this parameter set
            debug_folder = "debug_output"
            edges_path = os.path.join(debug_folder, f"debug_edges_canny_{low}_{high}.jpg")
            cv2.imwrite(edges_path, edges)
            print(f"  💾 Saved edges: {edges_path}")
        
    except Exception as e:
        print(f"❌ Error in detailed analysis: {e}")

def test_with_manual_parameters(image_path):
    """
    Test rotation detection with manually tuned parameters
    """
    print(f"\n🔧 Testing with manual parameter tuning...")
    
    try:
        image = Image.open(image_path)
        
        # Convert to OpenCV
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Try more aggressive edge detection for passport images
        edges = cv2.Canny(gray, 30, 100, apertureSize=3)
        
        # Try lower threshold for line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        
        if lines is not None:
            print(f"Manual tuning: Found {len(lines)} lines")
            
            # Filter lines by angle (keep mostly horizontal/vertical)
            filtered_angles = []
            for line in lines[:20]:  # Check more lines
                rho, theta = line[0]
                angle_deg = (theta - np.pi / 2) * 180 / np.pi
                
                # Keep angles that are reasonable for document rotation
                if abs(angle_deg) < 45:  # Within 45 degrees of horizontal
                    filtered_angles.append(angle_deg)
            
            if filtered_angles:
                median_angle = np.median(filtered_angles)
                print(f"Filtered angles: {[f'{a:.1f}' for a in filtered_angles[:10]]}")
                print(f"Recommended rotation: {median_angle:.2f}°")
                
                # Apply rotation if significant
                if abs(median_angle) > 0.5:
                    (h, w) = img_cv.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated_cv = cv2.warpAffine(img_cv, M, (w, h),
                                               flags=cv2.INTER_CUBIC,
                                               borderMode=cv2.BORDER_REPLICATE)
                    
                    # Save manually tuned result
                    debug_folder = "debug_output"
                    manual_path = os.path.join(debug_folder, f"manual_tuned_{median_angle:.1f}deg.jpg")
                    cv2.imwrite(manual_path, rotated_cv)
                    print(f"💾 Saved manually tuned result: {manual_path}")
            else:
                print("No suitable angles found after filtering")
        else:
            print("Manual tuning: No lines detected")
            
    except Exception as e:
        print(f"❌ Error in manual parameter testing: {e}")

if __name__ == "__main__":
    # Test with the specific image
    image_path = "temp/base64_20251223_161633_012205_4ad4e40d/rotation_debug/original_20251223_161633.jpg"
    
    print("🚀 Starting rotation detection debug...")
    result = debug_rotation_detection(image_path)
    
    if result:
        print("\n" + "="*60)
        print("MANUAL PARAMETER TESTING")
        print("="*60)
        test_with_manual_parameters(image_path)
    
    print("\n✅ Debug complete! Check the 'debug_output' folder for detailed analysis images.")