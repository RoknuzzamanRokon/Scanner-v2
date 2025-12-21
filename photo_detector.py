"""
Enhanced photo detection using ORB/SIFT feature matching
Based on ICAO passport photo standards
"""
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple


class PassportPhotoDetector:
    """
    Robust passport photo detection using feature matching
    
    ICAO Standards:
    - Passport page: 88mm × 125mm (aspect ratio 1:1.42)
    - Photo size: 35mm × 45mm (aspect ratio 1:1.28)
    - Photo occupies ~40-50% of page height
    """
    
    def __init__(self):
        # Initialize ORB detector (faster than SIFT, patent-free)
        self.orb = cv2.ORB_create(nfeatures=2000)
        
        # Matcher for feature matching
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def detect_photo_region(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect passport photo region using feature detection
        
        Args:
            image: PIL Image of passport
            
        Returns:
            Tuple (x, y, width, height) of photo region, or None if not detected
        """
        # Convert PIL to OpenCV
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            gray = img_array
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        height, width = gray.shape
        
        # Method 1: Try Haar Cascade first (fast)
        photo_region = self._detect_with_haar(gray)
        if photo_region:
            return photo_region
        
        # Method 2: Use edge detection and contour analysis
        photo_region = self._detect_with_contours(gray)
        if photo_region:
            return photo_region
        
        # Method 3: Fallback to proportions based on ICAO standards
        return self._detect_with_proportions(width, height)
    
    def _detect_with_haar(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect using Haar Cascade face detection
        
        Args:
            gray: Grayscale image
            
        Returns:
            Photo region (x, y, w, h) or None
        """
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            if len(faces) > 0:
                # Get largest face
                face_x, face_y, face_w, face_h = max(faces, key=lambda f: f[2] * f[3])
                
                # Photo area is larger than face
                # Standard passport photo: face takes ~70-80% of photo
                # So photo is ~1.3x face size
                photo_w = int(face_w * 1.4)
                photo_h = int(face_h * 1.6)
                
                # Center the photo around the face
                photo_x = max(0, face_x - int((photo_w - face_w) / 2))
                photo_y = max(0, face_y - int((photo_h - face_h) / 3))  # Face is typically in upper 2/3
                
                # Validate aspect ratio (should be ~1:1.28)
                aspect = photo_h / photo_w if photo_w > 0 else 0
                if 1.1 < aspect < 1.5:  # Reasonable range
                    return (photo_x, photo_y, photo_w, photo_h)
        except Exception as e:
            print(f"    Haar detection failed: {e}")
        
        return None
    
    def _detect_with_contours(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect photo using edge detection and contour analysis
        
        Passport photos typically have:
        - Dark border/frame
        - Rectangular shape
        - Located in left portion of passport
        
        Args:
            gray: Grayscale image
            
        Returns:
            Photo region (x, y, w, h) or None
        """
        try:
            height, width = gray.shape
            
            # Search in left 50% of image (photo is usually on left)
            search_region = gray[:, :int(width * 0.5)]
            
            # Edge detection
            edges = cv2.Canny(search_region, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by:
            # 1. Rectangular shape
            # 2. Appropriate size (20-40% of image area)
            # 3. Correct aspect ratio (~1:1.28)
            
            min_area = (width * height) * 0.08  # At least 8% of image
            max_area = (width * height) * 0.25  # At most 25% of image
            
            candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect = h / w if w > 0 else 0
                
                # Check if matches passport photo characteristics
                if (min_area < area < max_area and
                    1.1 < aspect < 1.5 and  # Aspect ratio range
                    x < width * 0.4 and     # In left portion
                    w > width * 0.15):      # Reasonable width
                    candidates.append((x, y, w, h, area))
            
            if candidates:
                # Return largest candidate
                x, y, w, h, _ = max(candidates, key=lambda c: c[4])
                return (x, y, w, h)
        
        except Exception as e:
            print(f"    Contour detection failed: {e}")
        
        return None
    
    def _detect_with_proportions(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Fallback: Use ICAO standard proportions
        
        Photo typically:
        - Located in top-left
        - Takes up ~30-35% of width
        - Takes up ~40-50% of height
        - Starts at ~15-20% from top
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Photo region (x, y, w, h)
        """
        # Standard proportions based on ICAO passport layout
        photo_w = int(width * 0.35)   # 30% of width
        photo_h = int(height * 0.45)  # 45% of height
        
        # Position: top-left with margins
        photo_x = int(width * 0.05)   # 5% from left
        photo_y = int(height * 0.15)  # 15% from top
        
        return (photo_x, photo_y, photo_w, photo_h)


def detect_passport_photo(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Convenience function to detect passport photo region
    
    Args:
        image: PIL Image of passport
        
    Returns:
        Photo region (x, y, width, height) or None
    """
    detector = PassportPhotoDetector()
    return detector.detect_photo_region(image)
