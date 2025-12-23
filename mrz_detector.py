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
        from passport_detector import save_crop_debug_images
        
        # Save detailed crop debug images
        save_crop_debug_images(image, regions, user_folder)
        
        # Also save the standard region debug image
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
