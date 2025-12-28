"""
Passport validation checker with TD3 compliance and confidence scoring
Enhanced with region detection for photo, MRZ, and passport page boundaries
"""
import re
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
from td3_validation_check import TD3_MRZ_RULES


def detect_all_regions(image: Image.Image, verbose: bool = True) -> Dict:
    """
    Detect all passport regions: photo, MRZ search zone, MRZ lines, and passport page
    
    Args:
        image: PIL Image object
        verbose: Print debug information
        
    Returns:
        Dictionary with detected regions:
        {
            'photo': (x, y, w, h) or None,
            'mrz_search_zone': (x, y, w, h) or None,
            'mrz_line1': (x, y, w, h) or None,
            'mrz_line2': (x, y, w, h) or None,
            'passport_page': (x, y, w, h) or None
        }
    """
    if verbose:
        print("  → Detecting passport regions (photo, MRZ, page boundaries)...")
    
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        gray = img_array
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        color_img = img_array.copy()
    
    height, width = gray.shape
    
    # Initialize results
    regions = {
        'photo': None,
        'mrz_search_zone': None,
        'mrz_line1': None,
        'mrz_line2': None,
        'passport_page': None
    }
    
    # Step 1: Detect photo region using face detection
    photo_region = detect_photo_region(gray, verbose)
    if photo_region:
        regions['photo'] = photo_region
        if verbose:
            print(f"    ✓ Photo detected: {photo_region}")
    
    # Step 2: Detect MRZ regions using text pattern detection
    mrz_regions = detect_mrz_regions(gray, verbose)
    regions.update(mrz_regions)
    
    # Step 3: Estimate passport page boundaries
    passport_page = estimate_passport_page_boundaries(
        gray, regions['photo'], regions['mrz_search_zone'], verbose
    )
    if passport_page:
        regions['passport_page'] = passport_page
        if verbose:
            print(f"    ✓ Passport page estimated: {passport_page}")
    
    return regions


def detect_photo_region(gray_image: np.ndarray, verbose: bool = True) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect passport photo region using face detection and geometric analysis
    
    Args:
        gray_image: Grayscale image as numpy array
        verbose: Print debug information
        
    Returns:
        (x, y, w, h) tuple or None if not found
    """
    try:
        # Use OpenCV face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Multiple detection parameters for better results
        detection_params = [
            {"scaleFactor": 1.1, "minNeighbors": 4, "minSize": (30, 30)},
            {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (40, 40)},
            {"scaleFactor": 1.2, "minNeighbors": 5, "minSize": (50, 50)}
        ]
        
        all_faces = []
        for params in detection_params:
            faces = face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=params["scaleFactor"],
                minNeighbors=params["minNeighbors"],
                minSize=params["minSize"]
            )
            if len(faces) > 0:
                all_faces.extend(faces)
        
        if len(all_faces) > 0:
            # Get the largest face (most likely the passport photo)
            largest_face = max(all_faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Expand to include typical passport photo margins
            padding = int(min(w, h) * 0.1)
            photo_x = max(0, x - padding)
            photo_y = max(0, y - padding)
            photo_w = min(gray_image.shape[1] - photo_x, w + 2 * padding)
            photo_h = min(gray_image.shape[0] - photo_y, h + 2 * padding)
            
            return (photo_x, photo_y, photo_w, photo_h)
        
        # Fallback: Look for rectangular regions in upper portion
        height, width = gray_image.shape
        upper_region = gray_image[:height//2, :width//3]  # Upper-left area
        
        # Edge detection for rectangular shapes
        edges = cv2.Canny(upper_region, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Minimum area for passport photo
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.6 < aspect_ratio < 1.4:  # Roughly square/rectangular
                    return (x, y, w, h)
        
        return None
        
    except Exception as e:
        if verbose:
            print(f"    ⚠ Photo detection error: {e}")
        return None


def detect_mrz_regions(gray_image: np.ndarray, verbose: bool = True) -> Dict:
    """
    Detect MRZ search zone and individual MRZ lines using multiple robust approaches
    
    Args:
        gray_image: Grayscale image as numpy array
        verbose: Print debug information
        
    Returns:
        Dictionary with MRZ regions
    """
    height, width = gray_image.shape
    
    # Focus on bottom portion where MRZ typically appears (more generous area)
    bottom_portion = gray_image[height//3:, :]  # Bottom 2/3 instead of 1/2
    
    # Method 1: Enhanced morphological operations for text line detection
    # Use multiple kernel sizes to catch different text patterns
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)),  # Thin horizontal
        cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2)),  # Medium horizontal
        cv2.getStructuringElement(cv2.MORPH_RECT, (70, 3)),  # Thick horizontal
    ]
    
    morph_results = []
    for kernel in kernels:
        morph = cv2.morphologyEx(bottom_portion, cv2.MORPH_CLOSE, kernel)
        morph_results.append(morph)
    
    # Combine morphological results
    combined_morph = np.maximum.reduce(morph_results)
    
    # Method 2: Enhanced edge detection with different thresholds
    edges_low = cv2.Canny(bottom_portion, 20, 60)   # Low threshold for faint text
    edges_med = cv2.Canny(bottom_portion, 50, 100)  # Medium threshold
    edges_high = cv2.Canny(bottom_portion, 80, 150) # High threshold for clear text
    
    # Combine edge results
    combined_edges = np.maximum.reduce([edges_low, edges_med, edges_high])
    
    # Method 3: Text-specific preprocessing
    # Apply adaptive threshold to handle varying lighting
    adaptive_thresh = cv2.adaptiveThreshold(
        bottom_portion, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Invert if needed (text should be dark on light background)
    if np.mean(adaptive_thresh) > 127:
        adaptive_thresh = cv2.bitwise_not(adaptive_thresh)
    
    # Apply morphological operations to connect text characters
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    text_morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, text_kernel)
    
    # Combine all methods
    final_combined = np.maximum.reduce([combined_morph, combined_edges, text_morph])
    
    # Find contours of potential text lines
    contours, _ = cv2.findContours(final_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for MRZ-like text lines with more lenient criteria
    mrz_candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # More lenient MRZ line criteria:
        # - Horizontal lines (at least 20% of image width)
        # - Reasonable height (between 5 and 60 pixels)
        # - Located in bottom portion
        # - Aspect ratio check (width >> height)
        aspect_ratio = w / h if h > 0 else 0
        
        if (w > width * 0.2 and          # At least 20% of width
            5 < h < 60 and               # Reasonable height range
            aspect_ratio > 3 and         # Much wider than tall
            y > height//6):              # In lower portion of search area
            
            # Adjust coordinates to full image
            full_y = y + height//3
            area = w * h
            mrz_candidates.append((x, full_y, w, h, area))
    
    # Method 4: Fallback - Look for any horizontal structures in bottom area
    if len(mrz_candidates) < 2:
        # Use Hough transform with more lenient parameters
        lines = cv2.HoughLinesP(
            combined_edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30,  # Lower threshold
            minLineLength=width//6,  # Shorter minimum length
            maxLineGap=20   # Allow larger gaps
        )
        
        if lines is not None:
            # Group lines by y-coordinate to find text lines
            line_groups = {}
            tolerance = 15  # Pixels tolerance for grouping
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 8:  # Nearly horizontal (more lenient)
                    y_avg = (y1 + y2) // 2 + height//3  # Adjust to full image
                    
                    # Find existing group or create new one
                    group_key = None
                    for existing_y in line_groups.keys():
                        if abs(y_avg - existing_y) < tolerance:
                            group_key = existing_y
                            break
                    
                    if group_key is None:
                        group_key = y_avg
                        line_groups[group_key] = []
                    
                    line_groups[group_key].append((min(x1, x2), max(x1, x2), y_avg))
            
            # Convert line groups to bounding boxes
            for y_group, lines_in_group in line_groups.items():
                if len(lines_in_group) >= 1:  # Even single lines can be MRZ
                    min_x = min(line[0] for line in lines_in_group)
                    max_x = max(line[1] for line in lines_in_group)
                    avg_y = sum(line[2] for line in lines_in_group) // len(lines_in_group)
                    
                    w = max_x - min_x
                    h = 25  # Estimated text height
                    
                    if w > width * 0.15:  # Even shorter lines acceptable
                        area = w * h
                        mrz_candidates.append((min_x, avg_y - h//2, w, h, area))
    
    # Method 5: Pattern-based detection for MRZ-like structures
    if len(mrz_candidates) < 2:
        # Look for repeating patterns that might indicate MRZ
        # Scan horizontal lines in bottom third
        for y in range(height*2//3, height, 5):  # Every 5 pixels
            if y >= height:
                break
                
            line = gray_image[y, :]
            
            # Look for alternating patterns (text/space/text/space)
            # Calculate variance - text lines have more variation
            line_variance = np.var(line)
            
            if line_variance > 100:  # Threshold for text-like variation
                # Find the extent of this text line
                start_x = 0
                end_x = width - 1
                
                # Try to find actual text boundaries
                for x in range(width//4):
                    if np.var(line[x:x+50]) > 50:  # Text detected
                        start_x = x
                        break
                
                for x in range(width*3//4, width-50):
                    if np.var(line[x:x+50]) > 50:  # Text detected
                        end_x = x + 50
                        break
                
                w = end_x - start_x
                if w > width * 0.2:  # Reasonable width
                    h = 20  # Estimated height
                    area = w * h
                    mrz_candidates.append((start_x, y - h//2, w, h, area))
    
    # Remove duplicates and overlapping candidates
    unique_candidates = []
    for candidate in mrz_candidates:
        x, y, w, h, area = candidate
        is_duplicate = False
        
        for existing in unique_candidates:
            ex, ey, ew, eh, _ = existing
            # Check for significant overlap
            overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
            overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
            
            if (overlap_y > h * 0.5 and overlap_x > w * 0.3):  # Significant overlap
                # Keep the larger one
                if area > existing[4]:
                    unique_candidates.remove(existing)
                else:
                    is_duplicate = True
                break
        
        if not is_duplicate:
            unique_candidates.append(candidate)
    
    # Sort by y-coordinate (top to bottom)
    unique_candidates.sort(key=lambda x: x[1])
    
    results = {
        'mrz_search_zone': None,
        'mrz_line1': None,
        'mrz_line2': None
    }
    
    if len(unique_candidates) >= 2:
        # Take the two bottom-most candidates as MRZ lines
        line1 = unique_candidates[-2][:4]  # Remove area from tuple
        line2 = unique_candidates[-1][:4]
        
        results['mrz_line1'] = line1
        results['mrz_line2'] = line2
        
        # Create MRZ search zone encompassing both lines
        min_x = min(line1[0], line2[0])
        min_y = line1[1]
        max_x = max(line1[0] + line1[2], line2[0] + line2[2])
        max_y = line2[1] + line2[3]
        
        # Add generous padding
        padding = 20
        search_x = max(0, min_x - padding)
        search_y = max(0, min_y - padding)
        search_w = min(width - search_x, max_x - min_x + 2 * padding)
        search_h = min(height - search_y, max_y - min_y + 2 * padding)
        
        results['mrz_search_zone'] = (search_x, search_y, search_w, search_h)
        
        if verbose:
            print(f"    ✓ MRZ Line 1: {line1}")
            print(f"    ✓ MRZ Line 2: {line2}")
            print(f"    ✓ MRZ Search Zone: {results['mrz_search_zone']}")
    
    elif len(unique_candidates) == 1:
        # Single MRZ line detected
        line = unique_candidates[0][:4]
        results['mrz_line2'] = line  # Assume it's the second line
        
        # Create search zone around single line with extra height for missing line
        padding_x = 25
        padding_y = 40  # Extra vertical padding to catch missing line
        search_x = max(0, line[0] - padding_x)
        search_y = max(0, line[1] - padding_y)
        search_w = min(width - search_x, line[2] + 2 * padding_x)
        search_h = min(height - search_y, line[3] + 2 * padding_y)
        
        results['mrz_search_zone'] = (search_x, search_y, search_w, search_h)
        
        if verbose:
            print(f"    ✓ Single MRZ line: {line}")
            print(f"    ✓ MRZ Search Zone: {results['mrz_search_zone']}")
    
    elif verbose:
        print(f"    ⚠ No MRZ lines detected (found {len(unique_candidates)} candidates)")
        if len(unique_candidates) > 0:
            print(f"    → Candidates were: {[c[:4] for c in unique_candidates]}")
    
    return results


def estimate_passport_page_boundaries(gray_image: np.ndarray, photo_region: Optional[Tuple], 
                                    mrz_region: Optional[Tuple], verbose: bool = True) -> Optional[Tuple[int, int, int, int]]:
    """
    Estimate passport page boundaries based on detected photo and MRZ regions
    
    Args:
        gray_image: Grayscale image as numpy array
        photo_region: (x, y, w, h) of photo or None
        mrz_region: (x, y, w, h) of MRZ search zone or None
        verbose: Print debug information
        
    Returns:
        (x, y, w, h) tuple for passport page boundaries or None
    """
    height, width = gray_image.shape
    
    if photo_region and mrz_region:
        # Use both photo and MRZ to estimate page boundaries
        photo_x, photo_y, photo_w, photo_h = photo_region
        mrz_x, mrz_y, mrz_w, mrz_h = mrz_region
        
        # Passport page typically extends beyond both regions
        page_left = 0  # Use full width - start from left edge
        page_top = max(0, photo_y - photo_h)  # Extend top by full photo height (increased from 30px)
        page_right = width  # Use full width - extend to right edge
        page_bottom = min(height, mrz_y + mrz_h + 20)
        
        page_w = page_right - page_left
        page_h = page_bottom - page_top
        
        return (page_left, page_top, page_w, page_h)
    
    elif photo_region:
        # Use photo region to estimate page
        photo_x, photo_y, photo_w, photo_h = photo_region
        
        # Estimate based on typical passport proportions
        page_left = 0  # Use full width - start from left edge
        page_top = max(0, photo_y - photo_h)  # Extend top by full photo height (increased from 20px)
        page_right = width  # Use full width - extend to right edge
        # Changed: Instead of going to bottom, stop at 20% above bottom (80% of image height)
        page_bottom = min(height, int(height * 0.8))  # Stop at 80% of image height (20% above bottom)
        
        page_w = page_right - page_left
        page_h = page_bottom - page_top
        
        return (page_left, page_top, page_w, page_h)
    
    elif mrz_region:
        # Use MRZ region to estimate page
        mrz_x, mrz_y, mrz_w, mrz_h = mrz_region
        
        # Estimate based on MRZ position (typically at bottom)
        page_left = max(0, mrz_x - 20)
        page_top = max(0, mrz_y - height//2)  # Assume MRZ is in bottom half
        page_right = min(width, mrz_x + mrz_w + 20)
        page_bottom = min(height, mrz_y + mrz_h + 20)
        
        page_w = page_right - page_left
        page_h = page_bottom - page_top
        
        return (page_left, page_top, page_w, page_h)
    
    # Fallback: return full image
    return (0, 0, width, height)


def save_crop_debug_images(image: Image.Image, regions: Dict, user_folder: str = None) -> None:
    """
    Save debug images showing crop areas before actual cropping
    
    Args:
        image: Original PIL Image
        regions: Dictionary of detected regions
        user_folder: User folder for saving debug images
    """
    try:
        import cv2
        import numpy as np
        from pathlib import Path
        from datetime import datetime
        
        # Create debug folder
        if user_folder:
            debug_folder = Path(user_folder) / "crop_debug"
        else:
            debug_folder = Path("temp") / "crop_debug"
        
        debug_folder.mkdir(parents=True, exist_ok=True)
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            debug_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            debug_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Create overlay image for transparency effects
        overlay = debug_img.copy()
        
        # Color scheme for different crop areas
        colors = {
            'photo': (0, 0, 255),           # Red
            'mrz_search_zone': (255, 255, 0),  # Cyan
            'mrz_line1': (255, 0, 255),    # Magenta
            'mrz_line2': (255, 0, 255),    # Magenta
            'passport_page': (0, 255, 0)   # Green
        }
        
        labels = {
            'photo': 'PHOTO CROP AREA',
            'mrz_search_zone': 'MRZ SEARCH ZONE',
            'mrz_line1': 'MRZ LINE 1',
            'mrz_line2': 'MRZ LINE 2',
            'passport_page': 'PASSPORT PAGE CROP'
        }
        
        # Draw crop areas with different visualization styles
        for region_name, region_coords in regions.items():
            if region_coords:
                x, y, w, h = region_coords
                color = colors.get(region_name, (128, 128, 128))
                label = labels.get(region_name, region_name)
                
                # Draw filled rectangle with transparency for crop area
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                
                # Draw border
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 3)
                
                # Add label with background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(debug_img, (x, y - 30), (x + label_size[0] + 10, y), color, -1)
                cv2.putText(debug_img, label, (x + 5, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add coordinates text
                coord_text = f"({x},{y}) {w}x{h}"
                cv2.putText(debug_img, coord_text, (x + 5, y + h - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Blend overlay with original for transparency effect
        alpha = 0.3  # Transparency level
        blended = cv2.addWeighted(debug_img, 1 - alpha, overlay, alpha, 0)
        
        # Save different debug views
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Original image with crop areas outlined
        outline_path = debug_folder / f"crop_areas_outline_{timestamp}.jpg"
        cv2.imwrite(str(outline_path), debug_img)
        print(f"  💾 Saved crop areas outline: {outline_path}")
        
        # 2. Original image with transparent crop areas
        transparent_path = debug_folder / f"crop_areas_transparent_{timestamp}.jpg"
        cv2.imwrite(str(transparent_path), blended)
        print(f"  💾 Saved transparent crop areas: {transparent_path}")
        
        # 3. Individual crop previews
        for region_name, region_coords in regions.items():
            if region_coords and region_name in ['photo', 'mrz_search_zone', 'passport_page']:
                x, y, w, h = region_coords
                
                # Ensure coordinates are valid
                x = max(0, min(x, img_array.shape[1] - 1))
                y = max(0, min(y, img_array.shape[0] - 1))
                w = min(w, img_array.shape[1] - x)
                h = min(h, img_array.shape[0] - y)
                
                if w > 0 and h > 0:
                    # Crop the region
                    cropped = img_array[y:y+h, x:x+w]
                    
                    if cropped.size > 0:
                        # Save individual crop preview
                        crop_path = debug_folder / f"crop_preview_{region_name}_{timestamp}.jpg"
                        
                        # Convert RGB to BGR for OpenCV saving
                        if len(cropped.shape) == 3:
                            cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                        else:
                            cropped_bgr = cropped
                        
                        cv2.imwrite(str(crop_path), cropped_bgr)
                        print(f"  💾 Saved {region_name} crop preview: {crop_path}")
        
        # 4. Create summary image with all information
        summary_img = debug_img.copy()
        
        # Add title
        cv2.putText(summary_img, "PASSPORT CROP AREAS DEBUG", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add legend
        legend_y = 60
        for region_name, region_coords in regions.items():
            if region_coords:
                color = colors.get(region_name, (128, 128, 128))
                label = labels.get(region_name, region_name)
                x, y, w, h = region_coords
                
                # Draw color box
                cv2.rectangle(summary_img, (10, legend_y), (30, legend_y + 20), color, -1)
                
                # Add text
                legend_text = f"{label}: ({x},{y}) {w}x{h}px"
                cv2.putText(summary_img, legend_text, (40, legend_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                legend_y += 30
        
        summary_path = debug_folder / f"crop_summary_{timestamp}.jpg"
        cv2.imwrite(str(summary_path), summary_img)
        print(f"  💾 Saved crop summary: {summary_path}")
        
        print(f"  📁 All debug images saved to: {debug_folder}")
        
    except Exception as e:
        print(f"  ⚠️ Failed to save crop debug images: {e}")


def save_region_debug_image(image: Image.Image, regions: Dict, save_path: str) -> None:
    """
    Save debug image with colored overlays showing detected regions
    
    Args:
        image: PIL Image object
        regions: Dictionary of detected regions
        save_path: Path to save debug image
    """
    try:
        import cv2
        import numpy as np
        from pathlib import Path
        
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            debug_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            debug_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Color scheme matching your example
        colors = {
            'photo': (0, 0, 255),           # Red
            'mrz_search_zone': (255, 255, 0),  # Cyan
            'mrz_line1': (255, 0, 255),    # Magenta
            'mrz_line2': (255, 0, 255),    # Magenta
            'passport_page': (0, 255, 0)   # Green
        }
        
        labels = {
            'photo': 'PHOTO',
            'mrz_search_zone': 'MRZ SEARCH ZONE',
            'mrz_line1': 'MRZ LINE 1 (P<)',
            'mrz_line2': 'MRZ LINE 2',
            'passport_page': 'Passport Page'
        }
        
        # Draw regions
        for region_name, region_coords in regions.items():
            if region_coords:
                x, y, w, h = region_coords
                color = colors.get(region_name, (128, 128, 128))
                
                # Draw rectangle
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                label = labels.get(region_name, region_name)
                cv2.putText(debug_img, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save debug image
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, debug_img)
        print(f"  💾 Saved region debug image: {save_path}")
        
    except Exception as e:
        print(f"  ⚠ Failed to save debug image: {e}")


def passport_validation_checker(mrz_text: str, verbose: bool = True) -> Dict:
    """
    STEP 3: Passport Validation Checker
    
    - MRZ text validation against TD3 standards
    - Confidence scoring (0.0 to 1.0)
    - Threshold-based validation (≥50% confidence)
    
    Args:
        mrz_text: MRZ text to validate (2 lines, 44 chars each)
        verbose: Print detailed logs
        
    Returns:
        Dictionary with validation results and confidence score
    """
    try:
        if verbose:
            print(f"  → Validating MRZ against TD3 standards...")
        
        if not mrz_text or not isinstance(mrz_text, str):
            if verbose:
                print(f"  ✗ Invalid MRZ text: Empty or non-string")
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "reason": "Empty or invalid MRZ text",
                "passport_data": {}
            }
        
        # Split into lines
        mrz_lines = mrz_text.strip().split('\n')
        
        if len(mrz_lines) != 2:
            if verbose:
                print(f"  ✗ Invalid MRZ format: Expected 2 lines, got {len(mrz_lines)}")
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "reason": f"Invalid line count: Expected 2, got {len(mrz_lines)}",
                "passport_data": {}
            }
        
        line1 = mrz_lines[0]
        line2 = mrz_lines[1]
        
        # Validate line lengths
        if len(line1) != 44 or len(line2) != 44:
            if verbose:
                print(f"  ✗ Invalid line lengths: Line1={len(line1)}, Line2={len(line2)} (expected 44)")
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "reason": f"Invalid line lengths: {len(line1)}, {len(line2)} (expected 44 each)",
                "passport_data": {}
            }
        
        # TD3 Validation Rules
        validation_score = 0
        max_score = 0
        issues = []
        
        # LINE 1 Validation
        line1_rules = TD3_MRZ_RULES["line1"]["fields"]
        
        # Document Type (pos 0-1)
        max_score += 1
        doc_type = line1[0:2]
        if re.match(r'^P[<A-Z]$', doc_type):
            validation_score += 1
        else:
            issues.append(f"Invalid document type: {doc_type}")
        
        # Country Code (pos 2-4)
        max_score += 1
        country_code = line1[2:5]
        if re.match(r'^[A-Z]{3}$', country_code.replace('<', 'X')):
            validation_score += 1
        else:
            issues.append(f"Invalid country code: {country_code}")
        
        # Name field (pos 5-43) - should contain at least one name
        max_score += 1
        name_field = line1[5:44]
        if '<<' in name_field and any(c.isalpha() for c in name_field):
            validation_score += 1
        else:
            issues.append(f"Invalid name field format")
        
        # LINE 2 Validation
        line2_rules = TD3_MRZ_RULES["line2"]["fields"]
        
        # Passport Number (pos 0-8)
        max_score += 1
        passport_num = line2[0:9].replace('<', '')
        if passport_num:  # Can be empty per TD3 standards
            validation_score += 1
        
        # Nationality (pos 10-12)
        max_score += 1
        nationality = line2[10:13]
        if re.match(r'^[A-Z]{3}$', nationality.replace('<', 'X')):
            validation_score += 1
        else:
            issues.append(f"Invalid nationality: {nationality}")
        
        # Date of Birth (pos 13-18)
        max_score += 1
        dob = line2[13:19]
        if re.match(r'^\\d{6}$', dob):
            validation_score += 1
        else:
            issues.append(f"Invalid date of birth: {dob}")
        
        # Sex (pos 20)
        max_score += 1
        sex = line2[20]
        
        # Normalize sex field (handle 0->M, 1->F mapping)
        from sex_field_normalizer import normalize_sex_field
        normalized_sex = normalize_sex_field(sex)
        
        if normalized_sex in ['M', 'F', 'X', '<']:
            validation_score += 1
        else:
            issues.append(f"Invalid sex field: {sex}")
        
        # Expiry Date (pos 21-26)
        max_score += 1
        expiry = line2[21:27]
        if re.match(r'^\\d{6}$', expiry):
            validation_score += 1
        else:
            issues.append(f"Invalid expiry date: {expiry}")
        
        # Calculate confidence score
        confidence = validation_score / max_score if max_score > 0 else 0.0
        
        # Determine if valid (≥50% confidence)
        is_valid = confidence >= 0.5
        
        if verbose:
            print(f"  → Validation Score: {validation_score}/{max_score}")
            print(f"  → Confidence: {confidence*100:.1f}%")
            print(f"  → Valid: {is_valid}")
            if issues:
                print(f"  → Issues found:")
                for issue in issues:
                    print(f"    - {issue}")
        
        # Extract basic passport data for return
        passport_data = {}
        if is_valid:
            # Parse name field
            name_parts = name_field.split('<<')
            surname = name_parts[0].replace('<', ' ').strip() if len(name_parts) > 0 else ""
            given_names = name_parts[1].replace('<', ' ').strip() if len(name_parts) > 1 else ""
            
            # Normalize sex field (convert 0->M, 1->F, etc.)
            from sex_field_normalizer import normalize_sex_field
            normalized_sex = normalize_sex_field(sex)
            
            passport_data = {
                "document_type": doc_type[0] if doc_type else "P",
                "country_code": country_code,
                "surname": surname,
                "given_names": given_names,
                "passport_number": passport_num,
                "nationality": nationality,
                "date_of_birth": dob,
                "sex": normalized_sex,
                "expiry_date": expiry,
                "personal_number": line2[28:42].replace('<', '').strip()
            }
        
        return {
            "is_valid": is_valid,
            "confidence_score": confidence,
            "reason": " | ".join(issues) if issues else "MRZ validation passed",
            "passport_data": passport_data
        }
    
    except Exception as e:
        import traceback
        if verbose:
            print(f"  ✗ Validation error: {e}")
            traceback.print_exc()
        
        return {
            "is_valid": False,
            "confidence_score": 0.0,
            "reason": f"Validation error: {str(e)}",
            "passport_data": {}
        }


def is_passport_image(text: str, verbose: bool = False) -> bool:
    """
    Check if image contains passport-like content
    
    Rules:
    - Contains '<' symbol more than 5 times (MRZ indicator)
    - Contains 'Passport' or 'PASSPORT' keyword
    - Contains country codes or MRZ patterns
    
    Args:
        text: OCR extracted text from image
        verbose: Print detailed logs
        
    Returns:
        True if likely a passport image
    """
    if not text:
        return False
    
    # Rule 1: Check for MRZ indicators (< symbols)
    angle_bracket_count = text.count('<')
    if angle_bracket_count > 5:
        if verbose:
            print(f"  ✓ MRZ detected: {angle_bracket_count} '<' symbols found")
        return True
    
    # Rule 2: Check for passport keywords
    passport_keywords = ['passport', 'PASSPORT', 'Passeport', 'Passaporto', 'Pasaporte', 'Reisepass']
    for keyword in passport_keywords:
        if keyword in text:
            if verbose:
                print(f"  ✓ Passport keyword found: {keyword}")
            return True
    
    # Rule 3: Check for MRZ pattern (lines starting with P<)
    if re.search(r'P<[A-Z]{3}', text):
        if verbose:
            print(f"  ✓ MRZ pattern detected (P<XXX)")
        return True
    
    if verbose:
        print(f"  ✗ No passport indicators found")
    
    return False