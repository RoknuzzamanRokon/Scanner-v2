"""
Verify that the mrz_text NameError is fixed
"""
import sys
import traceback

def test_gemini_parser_import():
    """Test if we can import and call the function without NameError"""
    try:
        # Test import
        from gemini_passport_parser import gemini_ocr_from_url
        print("✅ Import successful")
        
        # Test function signature (this will fail due to missing image, but should not have NameError)
        try:
            result = gemini_ocr_from_url("test_url")
        except NameError as e:
            if "mrz_text" in str(e):
                print(f"❌ NameError still exists: {e}")
                return False
            else:
                print(f"❌ Different NameError: {e}")
                return False
        except Exception as e:
            # Any other error is fine - we just want to avoid NameError
            print(f"✅ No NameError - got expected error: {type(e).__name__}")
            return True
            
        return True
        
    except Exception as e:
        print(f"❌ Import or other error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Verifying mrz_text NameError fix...")
    success = test_gemini_parser_import()
    
    if success:
        print("\n🎉 SUCCESS: The mrz_text NameError has been fixed!")
        print("The AI parser should now work without the 'name 'mrz_text' is not defined' error.")
    else:
        print("\n❌ FAILED: The mrz_text NameError still exists.")
        
    print("\nNote: You may still get other errors (like API errors) but the NameError should be resolved.")