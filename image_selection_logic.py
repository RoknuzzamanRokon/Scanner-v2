"""
Image selection logic for crop_debug folder
Priority: corrected_*.jpg > original_*.jpg > direct original image
"""
import os
import glob
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple


def find_best_image_for_crop_debug(user_folder: str, original_image: Image.Image) -> Tuple[Image.Image, str]:
    """
    Find the best image to use for crop_debug based on priority:
    1. First check rotation_debug/corrected_*.jpg
    2. If not found, check rotation_debug/original_*.jpg  
    3. If not found, use direct original image
    
    Args:
        user_folder: Base user folder path (e.g., "temp/base64_something/")
        original_image: The direct original PIL Image as fallback
        
    Returns:
        Tuple of (selected_image, source_description)
    """
    
    if not user_folder:
        return original_image, "direct_original (no user folder)"
    
    user_path = Path(user_folder)
    rotation_debug_path = user_path / "rotation_debug"
    
    print(f"  🔍 Searching for best image in: {user_folder}")
    
    # Priority 1: Check for corrected_*.jpg in rotation_debug/
    if rotation_debug_path.exists():
        corrected_pattern = str(rotation_debug_path / "corrected_*.jpg")
        corrected_files = glob.glob(corrected_pattern)
        
        if corrected_files:
            # Use the most recent corrected file
            corrected_file = max(corrected_files, key=os.path.getmtime)
            try:
                corrected_image = Image.open(corrected_file)
                print(f"  ✅ Found corrected image: {Path(corrected_file).name}")
                return corrected_image, f"rotation_corrected ({Path(corrected_file).name})"
            except Exception as e:
                print(f"  ⚠ Failed to load corrected image {corrected_file}: {e}")
        else:
            print(f"  ℹ No corrected_*.jpg found in rotation_debug/")
    else:
        print(f"  ℹ rotation_debug/ folder not found")
    
    # Priority 2: Check for original_*.jpg in rotation_debug/
    if rotation_debug_path.exists():
        original_pattern = str(rotation_debug_path / "original_*.jpg")
        original_files = glob.glob(original_pattern)
        
        if original_files:
            # Use the most recent original file
            original_file = max(original_files, key=os.path.getmtime)
            try:
                rotation_original_image = Image.open(original_file)
                print(f"  ✅ Found rotation debug original: {Path(original_file).name}")
                return rotation_original_image, f"rotation_original ({Path(original_file).name})"
            except Exception as e:
                print(f"  ⚠ Failed to load rotation original {original_file}: {e}")
        else:
            print(f"  ℹ No original_*.jpg found in rotation_debug/")
    
    # Priority 3: Use direct original image
    print(f"  ✅ Using direct original image as fallback")
    return original_image, "direct_original (fallback)"


def setup_crop_debug_with_best_image(user_folder: str, original_image: Image.Image) -> Tuple[Image.Image, str, Path]:
    """
    Set up crop_debug folder with the best available image
    
    Args:
        user_folder: Base user folder path
        original_image: The direct original PIL Image as fallback
        
    Returns:
        Tuple of (selected_image, source_description, crop_debug_path)
    """
    
    # Find the best image to use
    selected_image, source_description = find_best_image_for_crop_debug(user_folder, original_image)
    
    # Create crop_debug folder
    if user_folder:
        crop_debug_path = Path(user_folder) / "crop_debug"
    else:
        crop_debug_path = Path("temp") / "crop_debug"
    
    crop_debug_path.mkdir(parents=True, exist_ok=True)
    
    # Save the selected image as the base for crop debug
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_image_path = crop_debug_path / f"base_image_{timestamp}.jpg"
    
    try:
        selected_image.save(str(base_image_path), 'JPEG', quality=95)
        print(f"  💾 Saved base image for crop debug: {base_image_path.name}")
        print(f"  📋 Image source: {source_description}")
    except Exception as e:
        print(f"  ⚠ Failed to save base image: {e}")
    
    return selected_image, source_description, crop_debug_path


def test_image_selection_logic():
    """Test the image selection logic with current folder structure"""
    
    print("🧪 Testing image selection logic...")
    print("="*60)
    
    # Test with a sample user folder
    test_folders = [
        "temp/base64_20251223_172724_650068_522e0f9b",
        "temp/base64_20251223_172705_609847_abbc99fd", 
        "nonexistent_folder"
    ]
    
    # Create a dummy original image for testing
    dummy_image = Image.new('RGB', (100, 100), color='white')
    
    for test_folder in test_folders:
        print(f"\n📁 Testing folder: {test_folder}")
        print("-" * 40)
        
        if os.path.exists(test_folder):
            selected_image, source_desc = find_best_image_for_crop_debug(test_folder, dummy_image)
            print(f"  Result: {source_desc}")
            print(f"  Image size: {selected_image.size}")
        else:
            print(f"  ⚠ Folder does not exist")
            selected_image, source_desc = find_best_image_for_crop_debug(test_folder, dummy_image)
            print(f"  Result: {source_desc}")


if __name__ == "__main__":
    test_image_selection_logic()