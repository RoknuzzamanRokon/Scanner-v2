"""
Enhanced MRZ detection and cropping utilities
"""
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple


def detect_and_crop_data_section(image: Image.Image, save_debug: bool = False, image_url: str = "", user_folder = None) -> Optional[Image.Image]:
    """
    Unified detection: detect photo + MRZ, then crop passport page intelligently
    
    Process:
    1. Detect photo AND MRZ in one pass
    2. Use detections to estimate passport boundaries
    3. Crop full passport page
    4. Always save images even if detection partially fails
    
    Args:
        image: PIL Image object
        save_debug: Whether to save debug images
        image_url: URL of the image for AI fallback
        user_folder: User-specific folder path for saving debug images
        
    Returns:
        Cropped PIL Image of full passport page
    """

    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        gray = img_array
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        color_img = img_array.copy()
    
    height, width = gray.shape
    
    # Step 2: Unified detection - find photo AND MRZ together
    print("  → Detecting photo + MRZ (unified detection)...")
    
    from passport_detector import detect_all_regions, save_region_debug_image
    
    regions = detect_all_regions(image, verbose=True)
    photo = regions['photo']
    mrz = regions['mrz_search_zone']
    passport_page = regions['passport_page']
    
    # Save debug image with region overlays if requested
    if save_debug and user_folder:
        from pathlib import Path
        debug_path = Path(user_folder) / "region_detection_debug.jpg"
        save_region_debug_image(image, regions, str(debug_path))
    
    # Always ensure we have some kind of crop, even if detection fails
    if not passport_page:
        print("  → No passport page detected, using fallback crop...")
        # Fallback: use center 80% of image
        fallback_margin = int(min(width, height) * 0.1)
        passport_page = (fallback_margin, fallback_margin, 
                        width - 2*fallback_margin, height - 2*fallback_margin)
        print(f"  → Fallback crop: {passport_page}")
    
    if photo:
        photo_x, photo_y, photo_w, photo_h = photo
        print(f"  ✓ Photo detected: x={photo_x}-{photo_x+photo_w}, y={photo_y}-{photo_y+photo_h}")
    
    if mrz:
        mrz_x, mrz_y, mrz_w, mrz_h = mrz
        print(f"  ✓ MRZ detected: y={mrz_y}-{mrz_y+mrz_h}, height={mrz_h}px")
    
    # Always crop and save, using passport_page (detected or fallback)
    pass_x, pass_y, pass_w, pass_h = passport_page
    print(f"  ✓ Passport page: {pass_h}×{pass_w} pixels")
    
    # Ensure crop coordinates are valid
    pass_x = max(0, min(pass_x, width-1))
    pass_y = max(0, min(pass_y, height-1))
    pass_w = min(pass_w, width - pass_x)
    pass_h = min(pass_h, height - pass_y)
    
    # Crop passport page
    cropped = img_array[pass_y:pass_y+pass_h, pass_x:pass_x+pass_w]
    
    # Ensure we have a valid crop
    if cropped.size == 0:
        print("  ⚠ Crop failed, using full image...")
        cropped = img_array
        pass_x, pass_y, pass_w, pass_h = 0, 0, width, height
    
    # Store photo region for Step 3 (adjust coordinates to cropped image)
    if photo:
        # Adjust photo coordinates relative to cropped passport page
        adjusted_photo = (photo[0] - pass_x, photo[1] - pass_y, photo[2], photo[3])
        # Save for later use
        image._detected_photo_region = adjusted_photo
    
    # Always save images when save_debug is True
    if save_debug:
        # Save cropped passport page as imagedata_section_full.jpg
        passport_crop_pil = Image.fromarray(cropped)
        
        # Convert RGBA to RGB if needed (JPEG doesn't support alpha channel)
        if passport_crop_pil.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', passport_crop_pil.size, (255, 255, 255))
            if passport_crop_pil.mode == 'P':
                passport_crop_pil = passport_crop_pil.convert('RGBA')
            rgb_image.paste(passport_crop_pil, mask=passport_crop_pil.split()[-1] if passport_crop_pil.mode in ('RGBA', 'LA') else None)
            passport_crop_pil = rgb_image
        
        # Use user-specific folder if provided
        from pathlib import Path
        if user_folder:
            Path(user_folder).mkdir(parents=True, exist_ok=True)
            # Save as imagedata_section_full.jpg (the full data area)
            full_data_path = Path(user_folder) / 'imagedata_section_full.jpg'
        else:
            Path('temp').mkdir(exist_ok=True)
            full_data_path = Path('temp') / 'imagedata_section_full.jpg'
        
        try:
            passport_crop_pil.save(str(full_data_path), 'JPEG', quality=95)
            print(f"  ✓ Saved full data section: {full_data_path}")
        except Exception as e:
            print(f"  ⚠ Failed to save full data section: {e}")
        
        # Also save as passport_page_crop.jpg for backward compatibility
        try:
            if user_folder:
                save_path = Path(user_folder) / 'passport_page_crop.jpg'
            else:
                save_path = Path('temp') / 'passport_page_crop.jpg'
            passport_crop_pil.save(str(save_path), 'JPEG', quality=95)
            print(f"  ✓ Saved passport page crop: {save_path}")
        except Exception as e:
            print(f"  ⚠ Failed to save passport page crop: {e}")
        
        # Save MRZ zone if detected
        if mrz and mrz[3] > 10 and mrz[2] > 10:  # Valid MRZ dimensions
            try:
                print(f"  → Cropping MRZ search zone...")
                
                # Ensure MRZ coordinates are valid
                mrz_x = max(0, min(mrz[0], width-1))
                mrz_y = max(0, min(mrz[1], height-1))
                mrz_w = min(mrz[2], width - mrz_x)
                mrz_h = min(mrz[3], height - mrz_y)
                
                if mrz_w > 0 and mrz_h > 0:
                    mrz_zone_crop = img_array[mrz_y:mrz_y+mrz_h, mrz_x:mrz_x+mrz_w]
                    
                    if mrz_zone_crop.size > 0:
                        mrz_zone_pil = Image.fromarray(mrz_zone_crop)
                        
                        # Convert RGBA to RGB if needed
                        if mrz_zone_pil.mode in ('RGBA', 'LA', 'P'):
                            rgb_image = Image.new('RGB', mrz_zone_pil.size, (255, 255, 255))
                            if mrz_zone_pil.mode == 'P':
                                mrz_zone_pil = mrz_zone_pil.convert('RGBA')
                            rgb_image.paste(mrz_zone_pil, mask=mrz_zone_pil.split()[-1] if mrz_zone_pil.mode in ('RGBA', 'LA') else None)
                            mrz_zone_pil = rgb_image
                        
                        # Save MRZ zone crop
                        if user_folder:
                            mrz_save_path = Path(user_folder) / 'mrz_zone_crop.jpg'
                        else:
                            mrz_save_path = Path('temp') / 'mrz_zone_crop.jpg'
                        
                        mrz_zone_pil.save(str(mrz_save_path), 'JPEG', quality=95)
                        print(f"  ✓ Saved MRZ zone crop: {mrz_save_path}")
                        print(f"  ✓ Crop size: {mrz_w}×{mrz_h} pixels")
                    else:
                        print(f"  ⚠ MRZ crop resulted in empty image")
                else:
                    print(f"  ⚠ Invalid MRZ dimensions: {mrz_w}×{mrz_h}")
            except Exception as e:
                print(f"  ⚠ Failed to save MRZ zone crop: {e}")
        else:
            print(f"  → No valid MRZ zone to crop")
    
    # Always return the cropped image
    return Image.fromarray(cropped)
            if user_folder:
                save_path = Path(user_folder) / 'passport_page_crop.jpg'
            else:
                save_path = Path('temp') / 'passport_page_crop.jpg'
            passport_crop_pil.save(str(save_path))
            print(f"  ✓ Saved passport page crop: {save_path}")
            
            # Use pattern detection to find precise MRZ lines
            if photo and mrz:
                print(f"  → Cropping full MRZ search zone (below photo)...")
                
                # Check if MRZ zone has valid dimensions
                if mrz[3] > 10 and mrz[2] > 10:  # Minimum 10px height and width
                    # Always crop the FULL MRZ zone (entire area below photo)
                    # This is what was detected by the unified detector
                    mrz_zone_crop = img_array[mrz[1]:mrz[1]+mrz[3], mrz[0]:mrz[0]+mrz[2]]
                    mrz_zone_pil = Image.fromarray(mrz_zone_crop)
                    
                    # Convert RGBA to RGB if needed (JPEG doesn't support alpha channel)
                    if mrz_zone_pil.mode in ('RGBA', 'LA', 'P'):
                        rgb_image = Image.new('RGB', mrz_zone_pil.size, (255, 255, 255))
                        if mrz_zone_pil.mode == 'P':
                            mrz_zone_pil = mrz_zone_pil.convert('RGBA')
                        rgb_image.paste(mrz_zone_pil, mask=mrz_zone_pil.split()[-1] if mrz_zone_pil.mode in ('RGBA', 'LA') else None)
                        mrz_zone_pil = rgb_image
                    
                    # Use user-specific folder if provided
                    from pathlib import Path
                    if user_folder:
                        mrz_save_path = Path(user_folder) / 'mrz_zone_crop.jpg'
                    else:
                        mrz_save_path = Path('temp') / 'mrz_zone_crop.jpg'
                    mrz_zone_pil.save(str(mrz_save_path))
                    
                    print(f"  ✓ Saved MRZ zone crop: {mrz_save_path}")
                    print(f"  ✓ Crop size: {mrz[2]}×{mrz[3]} pixels (full width, full below photo)")
                else:
                    print(f"  ⚠ MRZ zone too small ({mrz[2]}×{mrz[3]} px), skipping crop")
                    print(f"  → Will use full-image OCR fallback")
                
                # Also try pattern detection for visualization (but don't use it for crop)
                try:
                    from mrz_pattern_detector import detect_mrz_zone_by_pattern
                    
                    # Detect MRZ by "P<" pattern - only for debug visualization
                    mrz_pattern_result = detect_mrz_zone_by_pattern(
                        image, 
                        photo_region=photo,
                        save_debug=True,  # This saves mrz_pattern_debug.jpg
                        image_url=image_url,
                        user_folder=user_folder
                    )
                    
                    if mrz_pattern_result:
                        pattern_x, pattern_y, pattern_w, pattern_h = mrz_pattern_result
                        print(f"  ✓ Pattern detection found MRZ lines: y={pattern_y}-{pattern_y+pattern_h}, height={pattern_h}px")
                    else:
                        print(f"  → Pattern detection failed (for visualization only)")
                except Exception as e:
                    print(f"  ⚠ Pattern detection error: {e}")
            
            # Draw all detections on the ORIGINAL IMAGE
            debug_img = color_img.copy()
            
            # Get individual MRZ line coordinates for detailed visualization
            mrz_lines_data = None
            if photo:
                try:
                    from mrz_pattern_detector import get_mrz_line_coordinates
                    mrz_lines_data = get_mrz_line_coordinates(image, photo_region=photo)
                    
                    # If pattern detection failed but we have MRZ zone, create estimated lines
                    if (not mrz_lines_data or not mrz_lines_data['line1'] or not mrz_lines_data['line2']) and mrz:
                        print(f"  → Pattern detection failed, using estimated MRZ line positions...")
                        # Estimate 2 equal lines within MRZ zone
                        mrz_line_h = int(mrz_h / 2) - 5  # Half height minus gap
                        
                        mrz_lines_data = {
                            'line1': (mrz_x, mrz_y + 5, mrz_w, mrz_line_h),  # Top half
                            'line2': (mrz_x, mrz_y + mrz_line_h + 10, mrz_w, mrz_line_h),  # Bottom half
                            'mrz_zone': (mrz_x, mrz_y, mrz_w, mrz_h)
                        }
                        print(f"  ✓ Estimated Line 1: y={mrz_y + 5}, h={mrz_line_h}px")
                        print(f"  ✓ Estimated Line 2: y={mrz_y + mrz_line_h + 10}, h={mrz_line_h}px")
                        
                except Exception as e:
                    print(f"  ⚠ Could not get MRZ line coordinates: {e}")
            
            if photo:
                # 1. Draw PHOTO box (BLUE)
                cv2.rectangle(debug_img, (photo_x, photo_y), (photo_x+photo_w, photo_y+photo_h), (255, 0, 0), 3)
                cv2.putText(debug_img, "PHOTO", (photo_x + 5, photo_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # 2. Draw MRZ SEARCH ZONE (YELLOW) - FULL area below photo
                mrz_search_start = photo_y + photo_h
                mrz_search_end = height  # To end of image
                
                # Draw semi-transparent yellow overlay
                overlay = debug_img.copy()
                cv2.rectangle(overlay, (0, mrz_search_start), (width, mrz_search_end), (0, 255, 255), -1)
                cv2.addWeighted(overlay, 0.15, debug_img, 0.85, 0, debug_img)
                
                # Draw yellow border
                cv2.rectangle(debug_img, (0, mrz_search_start), (width, mrz_search_end), (0, 255, 255), 3)
                cv2.putText(debug_img, "MRZ SEARCH ZONE (FULL below photo)", (10, mrz_search_start + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 3. Draw individual MRZ lines if detected (PURPLE colors)
            if mrz_lines_data and mrz_lines_data['line1'] and mrz_lines_data['line2']:
                line1_x, line1_y, line1_w, line1_h = mrz_lines_data['line1']
                line2_x, line2_y, line2_w, line2_h = mrz_lines_data['line2']
                
                # Draw Line 1 (P<) in MAGENTA
                cv2.rectangle(debug_img, (line1_x, line1_y), (line1_x+line1_w, line1_y+line1_h), (255, 0, 255), 3)
                cv2.putText(debug_img, "MRZ LINE 1 (P<)", (line1_x + 5, line1_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Draw Line 2 in PURPLE (darker)
                cv2.rectangle(debug_img, (line2_x, line2_y), (line2_x+line2_w, line2_y+line2_h), (128, 0, 128), 3)
                cv2.putText(debug_img, "MRZ LINE 2", (line2_x + 5, line2_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
            
            if mrz:
                # 4. Draw detected MRZ region (RED) - overall zone
                cv2.rectangle(debug_img, (mrz_x, mrz_y), (mrz_x+mrz_w, mrz_y+mrz_h), (0, 0, 255), 2)
                cv2.putText(debug_img, f"MRZ ZONE ({mrz_h}px)", (mrz_x + 5, mrz_y + mrz_h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 5. Draw PASSPORT PAGE boundary (GREEN)
            cv2.rectangle(debug_img, (pass_x, pass_y), (pass_x+pass_w, pass_y+pass_h), (0, 255, 0), 4)
            cv2.putText(debug_img, f"PASSPORT PAGE ({pass_w}x{pass_h}px)", (pass_x + 10, pass_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
            
            # Add legend
            legend_y = 30
            cv2.putText(debug_img, "LEGEND:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(debug_img, (10, legend_y + 10), (30, legend_y + 30), (255, 0, 0), -1)
            cv2.putText(debug_img, "= Photo", (35, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.rectangle(debug_img, (10, legend_y + 40), (30, legend_y + 60), (0, 255, 255), -1)
            cv2.putText(debug_img, "= MRZ Search Zone", (35, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.rectangle(debug_img, (10, legend_y + 70), (30, legend_y + 90), (255, 0, 255), -1)
            cv2.putText(debug_img, "= MRZ Line 1 (P<)", (35, legend_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.rectangle(debug_img, (10, legend_y + 100), (30, legend_y + 120), (128, 0, 128), -1)
            cv2.putText(debug_img, "= MRZ Line 2", (35, legend_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.rectangle(debug_img, (10, legend_y + 130), (30, legend_y + 150), (0, 255, 0), -1)
            cv2.putText(debug_img, "= Passport Page", (35, legend_y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            
            debug_pil = Image.fromarray(debug_img)
            
            # Convert RGBA to RGB if needed (JPEG doesn't support alpha channel)
            if debug_pil.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', debug_pil.size, (255, 255, 255))
                if debug_pil.mode == 'P':
                    debug_pil = debug_pil.convert('RGBA')
                rgb_image.paste(debug_pil, mask=debug_pil.split()[-1] if debug_pil.mode in ('RGBA', 'LA') else None)
                debug_pil = rgb_image
            
            # Use user-specific folder if provided
            from pathlib import Path
            if user_folder:
                debug_save_path = Path(user_folder) / 'data_section_debug.jpg'
            else:
                debug_save_path = Path('temp') / 'data_section_debug.jpg'
            debug_pil.save(str(debug_save_path))
            print(f"  ✓ Saved debug visualization: {debug_save_path}")
        
        return Image.fromarray(cropped)
    
    # Fallback
    print("  → Using full image")
    return image


def detect_and_crop_mrz_region(image: Image.Image, photo_region: Optional[Tuple] = None, save_debug: bool = False, user_folder = None) -> Optional[Image.Image]:
    """
    Crop MRZ region - directly crops the search zone below photo (20% down)
    This is the same zone shown in yellow in the debug image
    
    Args:
        image: PIL Image object (cropped passport page from Step 2)
        photo_region: (x, y, w, h) of photo within this cropped image, or None
        save_debug: Whether to save debug images
        
    Returns:
        Cropped PIL Image containing MRZ search zone (20% below photo)
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        gray = img_array
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    height, width = gray.shape
    
    # If photo region is provided, crop 20% down from photo bottom
    if photo_region:
        photo_x, photo_y, photo_w, photo_h = photo_region
        
        # MRZ zone starts at photo bottom
        mrz_start_y = photo_y + photo_h
        
        # MRZ zone extends 20% of image height down
        mrz_height = int(height * 0.20)
        mrz_end_y = min(height, mrz_start_y + mrz_height)
        
        # Crop the MRZ zone (full width of this image)
        mrz_cropped = img_array[mrz_start_y:mrz_end_y, :]
        
        print(f"  → MRZ zone: 20% below photo (y={mrz_start_y}-{mrz_end_y}, height={mrz_end_y - mrz_start_y}px)")
        
        if save_debug:
            # Save debug image showing the crop
            debug_img = img_array.copy()
            if len(debug_img.shape) == 2:
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
            
            cv2.rectangle(debug_img, (0, mrz_start_y), (width, mrz_end_y), (0, 255, 255), 3)
            cv2.putText(debug_img, f"MRZ ZONE (20% below photo)", (10, mrz_start_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            debug_pil = Image.fromarray(debug_img)
            from pathlib import Path
            if user_folder:
                debug_save_path = Path(user_folder) / 'mrz_detection_debug.jpg'
            else:
                debug_save_path = Path('temp') / 'mrz_detection_debug.jpg'
            debug_pil.save(str(debug_save_path))
        
        return Image.fromarray(mrz_cropped)
    
    else:
        # Fallback: No photo detected, crop bottom 25% of image
        print("  → No photo info, using bottom 25% as MRZ zone")
        bottom_crop_y = int(height * 0.75)
        mrz_fallback = img_array[bottom_crop_y:, :]
        
        return Image.fromarray(mrz_fallback)
    
    # Calculate horizontal projection (sum of white pixels per row)
    horizontal_projection = np.sum(binary, axis=1)
    
    # Normalize projection
    if np.max(horizontal_projection) > 0:
        horizontal_projection = horizontal_projection / np.max(horizontal_projection)
    
    # Find rows with significant text (MRZ lines)
    # MRZ lines have high density of text (>30% of max)
    text_threshold = 0.30
    text_rows = np.where(horizontal_projection > text_threshold)[0]
    
    if len(text_rows) > 0:
        # Group consecutive rows into lines
        # MRZ lines are typically 20-35 pixels tall each
        lines = []
        current_line_start = text_rows[0]
        current_line_end = text_rows[0]
        
        for i in range(1, len(text_rows)):
            if text_rows[i] - text_rows[i-1] <= 3:  # Consecutive (allow small gaps)
                current_line_end = text_rows[i]
            else:
                # End of current line, start new line
                line_height = current_line_end - current_line_start
                if line_height >= 15:  # Minimum height for valid MRZ line
                    lines.append((current_line_start, current_line_end))
                current_line_start = text_rows[i]
                current_line_end = text_rows[i]
        
        # Add last line
        line_height = current_line_end - current_line_start
        if line_height >= 15:
            lines.append((current_line_start, current_line_end))
        
        # MRZ should have 2-4 lines (allow space above/below MRZ)
        if 2 <= len(lines) <= 4:
            # Take the last 2-3 lines (bottom-most = MRZ)
            if len(lines) == 4:
                mrz_lines = lines[-3:]  # Take last 3 lines if 4 found
            else:
                mrz_lines = lines[-2:]  # Take last 2 lines
            
            mrz_top = mrz_lines[0][0]
            mrz_bottom = mrz_lines[-1][1]
            
            # Add more padding for context (15 pixels top/bottom instead of 5)
            padding = 15
            mrz_top = max(0, mrz_top - padding)
            mrz_bottom = min(search_region.shape[0], mrz_bottom + padding)
            
            # Convert to full image coordinates
            mrz_top_full = search_start_y + mrz_top
            mrz_bottom_full = search_start_y + mrz_bottom
            
            # Crop MRZ region (full width)
            mrz_cropped = img_array[mrz_top_full:mrz_bottom_full, :]
            
            mrz_height = mrz_bottom_full - mrz_top_full
            
            print(f"  ✓ MRZ detected: {len(mrz_lines)} lines")
            print(f"  ✓ MRZ position: y={mrz_top_full}-{mrz_bottom_full}")
            print(f"  ✓ MRZ height: {mrz_height} pixels")
            
            # Validate height (should be 40-200 pixels for 2-3 lines with context)
            if 40 < mrz_height < 200:
                if save_debug:
                    # Save debug image
                    debug_img = img_array.copy()
                    if len(debug_img.shape) == 2:
                        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
                    
                    cv2.rectangle(debug_img, (0, mrz_top_full), (width, mrz_bottom_full), (0, 255, 0), 3)
                    cv2.putText(debug_img, f"MRZ ({len(mrz_lines)} lines)", (10, mrz_top_full - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    debug_pil = Image.fromarray(debug_img)
                    from pathlib import Path
                    if user_folder:
                        debug_save_path = Path(user_folder) / 'mrz_detection_debug.jpg'
                    else:
                        debug_save_path = Path('temp') / 'mrz_detection_debug.jpg'
                    debug_pil.save(str(debug_save_path))
                
                return Image.fromarray(mrz_cropped)
            else:
                print(f"  ⚠ Invalid MRZ height ({mrz_height}px), trying fallback...")
        else:
            print(f"  ⚠ Found {len(lines)} lines (need 2-3), trying fallback...")
    
    # Method 2: Fallback using contour detection
    print("  → Fallback: Using contour-based MRZ detection...")
    
    # Apply morphological operations to connect MRZ text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for wide horizontal  contours (MRZ lines)
    mrz_candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # MRZ lines: very wide (>50% width), short (15-40px), high aspect ratio
        if w > width * 0.50 and 15 < h < 40 and aspect_ratio > 10:
            mrz_candidates.append((x, y, w, h))
    
    # Sort by y position and take bottom 2-3
    if len(mrz_candidates) >= 2:
        mrz_candidates = sorted(mrz_candidates, key=lambda c: c[1])
        mrz_lines = mrz_candidates[-2:]  # Take last 2 lines
        
        min_y = min(c[1] for c in mrz_lines)
        max_y = max(c[1] + c[3] for c in mrz_lines)
        
        # Minimal padding
        padding = 5
        min_y = max(0, min_y - padding)
        max_y = min(search_region.shape[0], max_y + padding)
        
        # Convert to full coordinates
        min_y_full = search_start_y + min_y
        max_y_full = search_start_y + max_y
        
        mrz_cropped = img_array[min_y_full:max_y_full, :]
        
        print(f"  ✓ MRZ detected via contours: {len(mrz_lines)} lines")
        print(f"  ✓ MRZ height: {max_y_full - min_y_full} pixels")
        
        return Image.fromarray(mrz_cropped)
    
    # Method 3: Final fallback - crop bottom 15% of passport
    print("  → Final fallback: Using bottom 15% of passport")
    bottom_crop_y = int(height * 0.85)
    mrz_fallback = img_array[bottom_crop_y:, :]
    
    return Image.fromarray(mrz_fallback)


def enhance_mrz_image(image: Image.Image) -> Image.Image:
    """
    Enhance MRZ region image for better OCR accuracy
    
    Args:
        image: PIL Image of MRZ region
        
    Returns:
        Enhanced PIL Image
    """
    # Convert to OpenCV
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Resize to improve OCR (make text larger)
    height, width = gray.shape
    scale_factor = 2.0
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(resized)
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    
    # Apply adaptive threshold for better text clarity
    binary = cv2.adaptiveThreshold(
        contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Invert if needed (text should be dark on light background)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    
    return Image.fromarray(binary)


def enhance_data_section(image: Image.Image, save_debug: bool = False) -> Image.Image:
    """
    Enhance data section image for better OCR accuracy
    Applies: 2x scaling, denoising, CLAHE contrast, binary conversion
    
    Args:
        image: PIL Image of data section
        save_debug: Whether to save debug images
        
    Returns:
        Enhanced PIL Image with black text on white background
    """
    # Convert to OpenCV
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Step 1: Scale up 2x for better text recognition
    height, width = gray.shape
    scale_factor = 2.0
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    print(f"  Enhanced: Scaled from {width}x{height} to {new_width}x{new_height} (2x)")
    
    # Step 2: Apply denoising
    denoised = cv2.fastNlMeansDenoising(resized, h=10)
    print(f"  Enhanced: Applied denoising")
    
    # Step 3: Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    print(f"  Enhanced: Applied CLAHE contrast enhancement")
    
    # Step 4: Convert to binary (black text on white background)
    binary = cv2.adaptiveThreshold(
        contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Ensure text is dark on light background
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    
    print(f"  Enhanced: Converted to binary (black text on white)")
    
    # Save debug if requested
    if save_debug:
        enhanced_pil = Image.fromarray(binary)
        # enhanced_pil.save('temp/data_section_enhanced.jpg')
        # print(f"  Enhanced: Saved debug image to temp/data_section_enhanced.jpg")
    
    return Image.fromarray(binary)


def preprocess_passport_for_mrz(image: Image.Image, auto_crop: bool = True) -> Image.Image:
    """
    Complete preprocessing pipeline for passport MRZ extraction
    
    Args:
        image: Original passport image
        auto_crop: Whether to auto-detect and crop MRZ region
        
    Returns:
        Preprocessed image ready for OCR
    """
    if auto_crop:
        # Detect and crop MRZ region
        cropped = detect_and_crop_mrz_region(image, save_debug=True)
        if cropped:
            # Enhance the cropped region
            enhanced = enhance_mrz_image(cropped)
            return enhanced
    
    # Fallback: use original preprocessing
    from utils import preprocess_for_mrz
    return preprocess_for_mrz(image)
