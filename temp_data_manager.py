"""
Temporary Data Manager for storing partial OCR results between methods
"""
import json
import os
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path


class TempDataManager:
    """
    Manages temporary JSON files for storing partial OCR results
    when validation threshold is not met
    """
    
    def __init__(self, user_id: str = None):
        """
        Initialize temp data manager
        
        Args:
            user_id: Unique user identifier
        """
        self.user_id = user_id or self._generate_user_id()
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.temp_file = self.temp_dir / f"partial_data_{self.user_id}.json"
    
    def _generate_user_id(self) -> str:
        """Generate a unique user ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"user_{timestamp}"
    
    def save_partial_data(self, method_name: str, passport_data: Dict, validation_results: Dict, mrz_text: str = "") -> bool:
        """
        Save partial OCR data when validation threshold is not met
        
        Args:
            method_name: Name of the OCR method (e.g., "PassportEye", "FastMRZ")
            passport_data: Extracted passport data
            validation_results: Field validation results
            mrz_text: MRZ text if available
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Analyze validation errors
            field_errors = {}
            for field, status in validation_results.get("field_results", {}).items():
                if status != "Valid":
                    # Map field names to error types
                    error_type = self._get_error_type(field, status, passport_data.get(field, ""))
                    field_errors[field] = error_type
            
            # Create the partial data structure
            partial_data = {
                "timestamp": datetime.now().isoformat(),
                "user_id": self.user_id,
                "from_method": method_name,
                "validation_summary": {
                    "valid_count": validation_results.get("valid_count", 0),
                    "total_count": validation_results.get("total_count", 10),
                    "threshold_met": validation_results.get("threshold_met", False)
                },
                "original_data": passport_data,
                "mrz_text": mrz_text,
                "get_error": field_errors,
                "field_details": validation_results.get("field_results", {}),
                "attempts": 1
            }
            
            # Check if file already exists and merge data
            if self.temp_file.exists():
                existing_data = self.load_partial_data()
                if existing_data:
                    # Update attempts count
                    partial_data["attempts"] = existing_data.get("attempts", 0) + 1
                    
                    # Keep the best data (highest valid count)
                    existing_valid_count = existing_data.get("validation_summary", {}).get("valid_count", 0)
                    current_valid_count = partial_data["validation_summary"]["valid_count"]
                    
                    if existing_valid_count >= current_valid_count:
                        # Keep existing data but update attempts
                        existing_data["attempts"] = partial_data["attempts"]
                        existing_data["last_method"] = method_name
                        partial_data = existing_data
                    else:
                        # Use new data but preserve attempt history
                        partial_data["previous_attempts"] = existing_data.get("previous_attempts", [])
                        partial_data["previous_attempts"].append({
                            "method": existing_data.get("from_method", "Unknown"),
                            "valid_count": existing_valid_count,
                            "timestamp": existing_data.get("timestamp", "")
                        })
            
            # Save to file
            with open(self.temp_file, 'w', encoding='utf-8') as f:
                json.dump(partial_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved partial data to: {self.temp_file}")
            print(f"   Method: {method_name}")
            print(f"   Valid fields: {validation_results.get('valid_count', 0)}/10")
            print(f"   Errors: {list(field_errors.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving partial data: {e}")
            return False
    
    def load_partial_data(self) -> Optional[Dict]:
        """
        Load existing partial data if available
        
        Returns:
            Dictionary with partial data or None if not found
        """
        try:
            if not self.temp_file.exists():
                return None
            
            with open(self.temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"📂 Loaded partial data from: {self.temp_file}")
            print(f"   From method: {data.get('from_method', 'Unknown')}")
            print(f"   Valid fields: {data.get('validation_summary', {}).get('valid_count', 0)}/10")
            print(f"   Attempts: {data.get('attempts', 1)}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading partial data: {e}")
            return None
    
    def has_partial_data(self) -> bool:
        """
        Check if partial data exists for this user
        
        Returns:
            True if partial data file exists, False otherwise
        """
        return self.temp_file.exists()
    
    def remove_partial_data(self) -> bool:
        """
        Remove partial data file (called when successful result is achieved)
        
        Returns:
            True if removed successfully, False otherwise
        """
        try:
            if self.temp_file.exists():
                self.temp_file.unlink()
                print(f"🗑️  Removed partial data file: {self.temp_file}")
                return True
            return False
            
        except Exception as e:
            print(f"❌ Error removing partial data: {e}")
            return False
    
    def get_improvement_suggestions(self, current_method: str) -> Dict:
        """
        Get suggestions for improving data based on previous attempts
        
        Args:
            current_method: Name of current OCR method
            
        Returns:
            Dictionary with improvement suggestions
        """
        partial_data = self.load_partial_data()
        if not partial_data:
            return {}
        
        suggestions = {
            "focus_fields": [],
            "known_good_fields": [],
            "problematic_fields": [],
            "previous_data": partial_data.get("original_data", {}),
            "previous_mrz": partial_data.get("mrz_text", "")
        }
        
        # Analyze field results
        field_results = partial_data.get("field_details", {})
        for field, status in field_results.items():
            if status == "Valid":
                suggestions["known_good_fields"].append(field)
            else:
                suggestions["problematic_fields"].append(field)
        
        # Focus on fields that were invalid in previous attempts
        suggestions["focus_fields"] = suggestions["problematic_fields"]
        
        return suggestions
    
    def _get_error_type(self, field_name: str, status: str, field_value: str) -> str:
        """
        Determine the type of error for a field
        
        Args:
            field_name: Name of the field
            status: Validation status
            field_value: Current field value
            
        Returns:
            Error type description
        """
        if status == "Invalid":
            if field_name in ["date_of_birth", "expiry_date"]:
                if not field_value or field_value == "":
                    return "missing_date"
                elif "00" in field_value or len(field_value) != 10:
                    return "invalid_format"
                else:
                    return "invalid_date"
            elif field_name == "sex":
                if not field_value or field_value in ["<", ""]:
                    return "missing_value"
                elif field_value not in ["M", "F", "X"]:
                    return "invalid_value"
            elif field_name in ["document_type", "issuing_country", "nationality"]:
                if not field_value or field_value == "":
                    return "missing_code"
                else:
                    return "invalid_code"
            elif field_name in ["surname", "given_names"]:
                if not field_value or field_value == "":
                    return "missing_name"
                else:
                    return "invalid_format"
            elif field_name == "passport_number":
                if not field_value or field_value == "":
                    return "missing_number"
                else:
                    return "invalid_format"
            else:
                return "validation_failed"
        elif status == "Missing":
            return "missing_field"
        else:
            return "unknown_error"
    
    @classmethod
    def cleanup_old_files(cls, max_age_hours: int = 24):
        """
        Clean up old partial data files
        
        Args:
            max_age_hours: Maximum age in hours before files are deleted
        """
        try:
            temp_dir = Path("temp")
            if not temp_dir.exists():
                return
            
            current_time = datetime.now().timestamp()
            max_age_seconds = max_age_hours * 3600
            
            for file in temp_dir.glob("partial_data_*.json"):
                file_age = current_time - file.stat().st_mtime
                if file_age > max_age_seconds:
                    file.unlink()
                    print(f"🧹 Cleaned up old partial data file: {file.name}")
                    
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")


# Convenience functions for easy integration
def save_partial_ocr_data(user_id: str, method_name: str, passport_data: Dict, validation_results: Dict, mrz_text: str = "") -> bool:
    """
    Convenience function to save partial OCR data
    """
    manager = TempDataManager(user_id)
    return manager.save_partial_data(method_name, passport_data, validation_results, mrz_text)


def load_partial_ocr_data(user_id: str) -> Optional[Dict]:
    """
    Convenience function to load partial OCR data
    """
    manager = TempDataManager(user_id)
    return manager.load_partial_data()


def remove_partial_ocr_data(user_id: str) -> bool:
    """
    Convenience function to remove partial OCR data
    """
    manager = TempDataManager(user_id)
    return manager.remove_partial_data()


def has_partial_ocr_data(user_id: str) -> bool:
    """
    Convenience function to check if partial OCR data exists
    """
    manager = TempDataManager(user_id)
    return manager.has_partial_data()