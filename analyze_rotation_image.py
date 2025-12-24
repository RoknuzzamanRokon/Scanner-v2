"""
Analyze the specific image that needs rotation
"""
import os
from PIL import Image
import cv2
import numpy as np

def analyze_specific_image(image_path):
    """Analyze the specific image that needs rotation"""
    
    print(f"🔍 Analyzing image: {image_path}")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    try:
        # Load image
        image = Image.open(image_path)
        print(f"✅ Image loaded: {image.size} {image.mode}")
        
        # Analyze image properties
        width, height = image.size
        aspect_ratio = width / height
        
        print(f"📐 Dimensions: {width}x{height}")
        print(f"📐 Aspect ratio: {aspect_ratio:.2f}")
        
        if aspect_ratio > 1.0:
            print(f"📐 Image is LANDSCAPE (wider than tall)")
        else:
            print(f"📐 Image is PORTRAIT (taller than wide)")
        
        # Convert to OpenCV for line analysis
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_percentage = (edge_pixels / total_pixels) * 100
        
        print(f"🔍 Edge detection: {edge_pixels} pixels ({edge_percentage:.2f}% of image)")
        
        # Line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is not None:
            print(f"📏 Detected {len(lines)} lines")
            
            # Analyze line angles
            angles = []
            for i in range(min(20, len(lines))):
                rho, theta = lines[i][0]
                angle_deg = (theta - np.pi / 2) * 180 / np.pi
                angles.append(angle_deg)
            
            # Categorize angles
            near_horizontal = [a for a in angles if abs(a) < 15]
            near_vertical = [a for a in angles if abs(abs(a) - 90) < 15]
            
            print(f"📏 Line analysis:")
            print(f"   Horizontal lines (±15°): {len(near_horizontal)}")
            print(f"   Vertical lines (±15°): {len(near_vertical)}")
            print(f"   Sample angles: {[f'{a:.1f}°' for a in angles[:10]]}")
            
            if len(near_vertical) > len(near_horizontal):
                print(f"🔄 ANALYSIS: More vertical lines detected - image likely needs rotation")
                
                # Determine rotation direction
                vertical_median = np.median(near_vertical) if near_vertical else 0
                if vertical_median < 0:
                    suggested_rotation = 90
                else:
                    suggested_rotation = -90
                    
                print(f"🔄 SUGGESTED: Rotate {suggested_rotation}° to correct orientation")
                
            elif len(near_horizontal) > len(near_vertical):
                print(f"✅ ANALYSIS: More horizontal lines - image appears correctly oriented")
            else:
                print(f"❓ ANALYSIS: Mixed line orientations - unclear if rotation needed")
        else:
            print(f"📏 No lines detected in image")
        
        return image
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return None

def test_with_specific_image():
    """Test with the specific image mentioned"""
    image_path = r"temp\base64_20251224_081844_099541_00fd35ca\rotation_debug\original_20251224_081844.jpg"
    
    image = analyze_specific_image(image_path)
    
    if image:
        print(f"\n" + "="*60)
        print("TESTING CURRENT ROTATION ALGORITHM")
        print("="*60)
        
        # Test current algorithm (should be disabled)
        from image_rotation_detector import detect_and_correct_rotation
        result = detect_and_correct_rotation(image, verbose=True)
        
        print(f"\n📊 CURRENT ALGORITHM RESULT:")
        print(f"   Rotation applied: {result['rotation_applied']}°")
        print(f"   Method: {result['method_used']}")
        
        if result['rotation_applied'] == 0:
            print(f"❌ PROBLEM: Algorithm didn't rotate the image (rotation is disabled)")
        else:
            print(f"✅ Algorithm applied rotation: {result['rotation_applied']}°")

if __name__ == "__main__":
    test_with_specific_image()