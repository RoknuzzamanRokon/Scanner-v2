"""
Main scanner function with multi-layered fallback system
Following the flow diagram: FastMRZ → PassportEye → EasyOCR → Tesseract → Validator → AI (if enabled)
"""
import time
from PIL import Image
from typing import Dict, Optional
from utils import download_image, decode_base64_image
from passportEye import validate_passport_with_PassportEye_fallback
from fastMRZ import validate_passport_with_fastmrz_fallback  
from passport_detector import passport_validation_checker
from gemini_passport_parser import gemini_ocr
from function_handler_switch import is_step_enabled, print_step_status


def scan_passport(
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    document_type: str = "passport",
    use_gemini: bool = True,
    step_config_override: Optional[Dict[str, bool]] = None
) -> Dict:
    """
    Scan passport image and extract MRZ data using multi-layered fallback system
    
    Flow (AI=ON):
    1. FastMRZ Fallback → Success? Return
    2. PassportEye Fallback → Success? Return
    3. EasyOCR Fallback → Success? Return
    4. Tesseract OCR Fallback → Success? Return
    5. Passport Validation → Valid? Continue
    6. AI Parser (Gemini) → Success? Return
    7. All failed → Error
    
    Flow (AI=OFF):
    1. FastMRZ Fallback → Success? Return
    2. PassportEye Fallback → Success? Return
    3. EasyOCR Fallback → Success? Return
    4. Tesseract OCR Fallback → Success? Return
    5. Passport Validation → Valid? Return
    6. All failed → Error
    
    Args:
        image_url: URL of the passport image
        image_base64: Base64 encoded passport image
        document_type: Type of document (passport, id_card, visa)
        use_gemini: Whether to use AI as final fallback
        
    Returns:
        Dictionary with extracted passport data
    """
    total_start_time = time.time()
    step_timings = {}
    working_process_step = {}
    
    try:
        # Step 0: Image Loading & Preprocessing
        print("\n" + "="*60)
        print("📄 PASSPORT SCANNER - MULTI-LAYERED FALLBACK SYSTEM")
        print("="*60)
        print(f"🔧 AI Mode: {'ON' if use_gemini else 'OFF'}")
        
        # Show step controller status
        print_step_status()
        
        step_start = time.time()
        if image_url:
            print(f"\n📥 Loading image from URL...")
            image = download_image(image_url)
            print(f"  ✓ Image loaded: {image.size} {image.mode}")
        elif image_base64:
            print(f"\n📥 Decoding base64 image...")
            image = decode_base64_image(image_base64)
            print(f"  ✓ Image decoded: {image.size} {image.mode}")
        else:
            return {
                "success": False,
                "passport_data": {},
                "mrz_text": "",
                "working_process_step": {},
                "step_timings": {},
                "total_time": "0.00s",
                "error": "No image provided. Please provide either image_url or image_base64",
                "validation_reason": "",
                "validation_confidence": ""
            }
        
        step_timings["image_loading"] = f"{time.time() - step_start:.2f}s"
        
        # Generate user_id and folder for temp files (needed for debug images)
        from utils import get_user_id_from_url, get_user_id_from_base64, create_user_temp_folder
        user_id = None
        user_folder = None
        user_folder_path = None
        
        if image_url:
            user_id = get_user_id_from_url(image_url)
        elif image_base64:
            user_id = get_user_id_from_base64(image_base64)
        
        if user_id:
            user_folder_path = create_user_temp_folder(user_id)
            user_folder = str(user_folder_path)
        
        # Initialize result variables to avoid undefined errors
        fastmrz_result = {"error": "Step not executed", "success": False}
        passporteye_result = {"error": "Step not executed", "success": False}
        easyocr_result = {"error": "Step not executed", "success": False}
        tesseract_result = {"error": "Step not executed", "success": False}
        ai_result = {"error": "Step not executed", "success": False}
        
        # STEP 1: FastMRZ Fallback Validation
        if is_step_enabled("STEP1", step_config_override):
            print("\n" + "-"*60)
            print("🔍 STEP 1: FastMRZ Fallback Validation")
            print("-"*60)
            
            step_start = time.time()
            fastmrz_result = validate_passport_with_fastmrz_fallback(image, verbose=True, user_id=user_id)
            step_timings["step1_fastmrz"] = f"{time.time() - step_start:.2f}s"
            working_process_step["step1_fastmrz"] = fastmrz_result.get("method_used", "FastMRZ")
            
            if fastmrz_result.get("success", False):
                print("\n✅ SUCCESS via FastMRZ (Early Exit)")
                
                # Remove validation failure file on success
                if user_id:
                    from utils import remove_validation_failures
                    remove_validation_failures(user_id)
                
                # Clean up user temp folder on success
                if user_folder_path:
                    from utils import cleanup_user_folder
                    cleanup_user_folder(user_folder_path)
                
                total_time = time.time() - total_start_time
                
                return {
                    "success": True,
                    "passport_data": {
                        "processVia": "FastMRZ", 
                        **fastmrz_result.get("passport_data", {})
                    },
                    "mrz_text": fastmrz_result.get("mrz_text", ""),
                    "working_process_step": working_process_step,
                    "step_timings": step_timings,
                    "total_time": f"{total_time:.2f}s",
                    "error": "",
                    "validation_reason": "",
                    "validation_confidence": ""
                }
            
            print(f"  ⚠ FastMRZ failed: {fastmrz_result.get('error', 'Unknown error')}")
        else:
            print("\n" + "-"*60)
            print("⏭ STEP 1: FastMRZ Fallback Validation - SKIPPED (DISABLED)")
            print("-"*60)
            fastmrz_result = {"error": "Step disabled", "success": False}
            step_timings["step1_fastmrz"] = "0.00s"
            working_process_step["step1_fastmrz"] = "Skipped (Disabled)"
        
        # STEP 2: PassportEye Fallback Validation
        if is_step_enabled("STEP2", step_config_override):
            print("\n" + "-"*60)
            print("🔍 STEP 2: PassportEye Fallback Validation")
            print("-"*60)
            
            step_start = time.time()
            passporteye_result = validate_passport_with_PassportEye_fallback(image, verbose=True, user_id=user_id)
            step_timings["step2_passporteye"] = f"{time.time() - step_start:.2f}s"
            working_process_step["step2_passporteye"] = passporteye_result.get("method_used", "PassportEye")
            
            if passporteye_result.get("success", False):
                print("\n✅ SUCCESS via PassportEye (Early Exit)")
                
                # Remove validation failure file on success
                if user_id:
                    from utils import remove_validation_failures
                    remove_validation_failures(user_id)
                
                # Clean up user temp folder on success
                if user_folder_path:
                    from utils import cleanup_user_folder
                    cleanup_user_folder(user_folder_path)
                
                total_time = time.time() - total_start_time
                
                return {
                    "success": True,
                    "passport_data": {
                        "processVia": "PassportEye",
                        **passporteye_result.get("passport_data", {})
                    },
                    "mrz_text": passporteye_result.get("mrz_text", ""),
                    "working_process_step": working_process_step,
                    "step_timings": step_timings,
                    "total_time": f"{total_time:.2f}s",
                    "error": "",
                    "validation_reason": "",
                    "validation_confidence": ""
                }
            
            print(f"  ⚠ PassportEye failed: {passporteye_result.get('error', 'Unknown error')}")
        else:
            print("\n" + "-"*60)
            print("⏭ STEP 2: PassportEye Fallback Validation - SKIPPED (DISABLED)")
            print("-"*60)
            passporteye_result = {"error": "Step disabled", "success": False}
            step_timings["step2_passporteye"] = "0.00s"
            working_process_step["step2_passporteye"] = "Skipped (Disabled)"
        


        # STEP 3: EasyOCR Fallback Validation
        if is_step_enabled("STEP3", step_config_override):
            print("\n" + "-"*60)
            print("🔍 STEP 3: EasyOCR Fallback Validation")
            print("-"*60)
            
            step_start = time.time()
            from easyOCR import validate_passport_with_easyocr_fallback
            easyocr_result = validate_passport_with_easyocr_fallback(image, verbose=True, user_folder=user_folder, user_id=user_id)
            step_timings["step3_easyocr"] = f"{time.time() - step_start:.2f}s"
            working_process_step["step3_easyocr"] = easyocr_result.get("method_used", "EasyOCR")
            
            if easyocr_result.get("success", False):
                print("\n✅ SUCCESS via EasyOCR (Early Exit)")
                
                # Remove validation failure file on success
                if user_id:
                    from utils import remove_validation_failures
                    remove_validation_failures(user_id)
                
                # Clean up user temp folder on success
                if user_folder_path:
                    from utils import cleanup_user_folder
                    cleanup_user_folder(user_folder_path)
                
                total_time = time.time() - total_start_time
                
                return {
                    "success": True,
                    "passport_data": {
                        "processVia": "EasyOCR",
                        **easyocr_result.get("passport_data", {})
                    },
                    "mrz_text": easyocr_result.get("mrz_text", ""),
                    "working_process_step": working_process_step,
                    "step_timings": step_timings,
                    "total_time": f"{total_time:.2f}s",
                    "error": "",
                    "validation_reason": "",
                    "validation_confidence": ""
                }
            
            print(f"  ⚠ EasyOCR failed: {easyocr_result.get('error', 'Unknown error')}")
        else:
            print("\n" + "-"*60)
            print("⏭ STEP 3: EasyOCR Fallback Validation - SKIPPED (DISABLED)")
            print("-"*60)
            easyocr_result = {"error": "Step disabled", "success": False}
            step_timings["step3_easyocr"] = "0.00s"
            working_process_step["step3_easyocr"] = "Skipped (Disabled)"
        
        # STEP 4: Tesseract OCR Fallback Validation
        if is_step_enabled("STEP4", step_config_override):
            print("\n" + "-"*60)
            print("🔍 STEP 4: Tesseract OCR Fallback Validation")
            print("-"*60)
            
            step_start = time.time()
            try:
                from tesseractOCR import validate_passport_with_tesseract_fallback
                tesseract_result = validate_passport_with_tesseract_fallback(image, verbose=True, user_id=user_id)
                step_timings["step4_tesseract"] = f"{time.time() - step_start:.2f}s"
                working_process_step["step4_tesseract"] = tesseract_result.get("method_used", "Tesseract")
                
                if tesseract_result.get("success", False):
                    print("\n✅ SUCCESS via Tesseract OCR (Early Exit)")
                    
                    # Remove validation failure file on success
                    if user_id:
                        from utils import remove_validation_failures
                        remove_validation_failures(user_id)
                    
                    # Clean up user temp folder on success
                    if user_folder_path:
                        from utils import cleanup_user_folder
                        cleanup_user_folder(user_folder_path)
                    
                    total_time = time.time() - total_start_time
                    
                    return {
                        "success": True,
                        "passport_data": {
                            "processVia": "Tesseract",
                            **tesseract_result.get("passport_data", {})
                        },
                        "mrz_text": tesseract_result.get("mrz_text", ""),
                        "working_process_step": working_process_step,
                        "step_timings": step_timings,
                        "total_time": f"{total_time:.2f}s",
                        "error": "",
                        "validation_reason": "",
                        "validation_confidence": ""
                    }
                
                print(f"  ⚠ Tesseract failed: {tesseract_result.get('error', 'Unknown error')}")
                
            except ImportError as e:
                print(f"  ⚠ Tesseract not available: {e}")
                tesseract_result = {"error": f"Tesseract not available: {e}"}
                step_timings["step4_tesseract"] = f"{time.time() - step_start:.2f}s"
                working_process_step["step4_tesseract"] = "Skipped (Import Error)"
            except Exception as e:
                print(f"  ⚠ Tesseract error: {e}")
                tesseract_result = {"error": f"Tesseract error: {e}"}
                step_timings["step4_tesseract"] = f"{time.time() - step_start:.2f}s"
                working_process_step["step4_tesseract"] = "Failed"
        else:
            print("\n" + "-"*60)
            print("⏭ STEP 4: Tesseract OCR Fallback Validation - SKIPPED (DISABLED)")
            print("-"*60)
            tesseract_result = {"error": "Step disabled", "success": False}
            step_timings["step4_tesseract"] = "0.00s"
            working_process_step["step4_tesseract"] = "Skipped (Disabled)"
        
        # STEP 5: Passport Validation Checker  
        if is_step_enabled("STEP5", step_config_override):
            print("\n" + "-"*60)
            print("🔍 STEP 5: Passport Validation Checker")
            print("-"*60)
            
            # Get MRZ text from previous attempts
            mrz_text = fastmrz_result.get("mrz_text", "") or passporteye_result.get("mrz_text", "") or easyocr_result.get("mrz_text", "") or tesseract_result.get("mrz_text", "")
            
            step_start = time.time()
            if mrz_text:
                validation_result = passport_validation_checker(mrz_text, verbose=True)
                step_timings["step5_validation"] = f"{time.time() - step_start:.2f}s"
                working_process_step["step5_validation"] = "TD3 Validation"
                
                confidence = validation_result.get("confidence_score", 0.0)
                is_valid = validation_result.get("is_valid", False)
                
                print(f"  → Validation Result: {is_valid}")
                print(f"  → Confidence Score: {confidence*100:.1f}%")
                
                if is_valid and confidence >= 0.5:
                    if not use_gemini:
                        # AI=OFF: Return success if validation passes
                        print("\n✅ SUCCESS via Validation (AI=OFF Mode)")
                        
                        # Clean up user temp folder on success
                        if user_folder_path:
                            from utils import cleanup_user_folder
                            cleanup_user_folder(user_folder_path)
                        
                        total_time = time.time() - total_start_time
                        
                        return {
                            "success": True,
                            "passport_data": {
                                "processVia": "EasyOCR",
                                **validation_result.get("passport_data", {})
                            },
                            "mrz_text": mrz_text,
                            "working_process_step": working_process_step,
                            "step_timings": step_timings,
                            "total_time": f"{total_time:.2f}s",
                            "error": "",
                            "validation_reason": validation_result.get("reason", ""),
                            "validation_confidence": f"{confidence*100:.1f}%"
                        }
                    else:
                        print(f"  → Validation passed but using AI for enhanced extraction...")
                else:
                    print(f"  ⚠ Validation failed: {validation_result.get('reason', 'Low confidence')}")
            else:
                step_timings["step5_validation"] = f"{time.time() - step_start:.2f}s"
                working_process_step["step5_validation"] = "Skipped (No MRZ)"
                print(f"  ⚠ No MRZ text available for validation")
        else:
            print("\n" + "-"*60)
            print("⏭ STEP 5: Passport Validation Checker - SKIPPED (DISABLED)")
            print("-"*60)
            step_timings["step5_validation"] = "0.00s"
            working_process_step["step5_validation"] = "Skipped (Disabled)"
        


        # STEP 6: ITTCheck Validation
        if is_step_enabled("STEP6", step_config_override):
            print("\n" + "-"*60)
            print("🔍 STEP 6: ITTCheck Validation")
            print("-"*60)
            
            step_start = time.time()
            try:
                from ittcheck import validate_passport_with_ittcheck
                ittcheck_result = validate_passport_with_ittcheck(image, verbose=True, user_id=user_id)
                

                
                step_timings["step6_ittcheck"] = f"{time.time() - step_start:.2f}s"
                working_process_step["step6_ittcheck"] = ittcheck_result.get("method_used", "ITTCheck")
                
                if ittcheck_result.get("success", False):
                    print("\n✅ SUCCESS via ITTCheck")
                    
                    # Remove validation failure file on success
                    if user_id:
                        from utils import remove_validation_failures
                        remove_validation_failures(user_id)
                    
                    # Clean up user temp folder on success
                    if user_folder_path:
                        from utils import cleanup_user_folder
                        cleanup_user_folder(user_folder_path)
                    
                    total_time = time.time() - total_start_time
                    
                    return {
                        "success": True,
                        "passport_data": {
                            "processVia": "ITTCheck",
                            **ittcheck_result.get("passport_data", {})
                        },
                        "mrz_text": ittcheck_result.get("mrz_text", ""),
                        "working_process_step": working_process_step,
                        "step_timings": step_timings,
                        "total_time": f"{total_time:.2f}s",
                        "error": "",
                        "validation_reason": "",
                        "validation_confidence": ""
                    }
                
                print(f"  ⚠ ITTCheck failed: {ittcheck_result.get('error', 'Unknown error')}")
                
            except ImportError as e:
                print(f"  ⚠ ITTCheck not available: {e}")
                ittcheck_result = {"error": f"ITTCheck not available: {e}"}
                step_timings["step6_ittcheck"] = f"{time.time() - step_start:.2f}s"
                working_process_step["step6_ittcheck"] = "Skipped (Import Error)"
            except Exception as e:
                print(f"  ⚠ ITTCheck error: {e}")
                ittcheck_result = {"error": f"ITTCheck processing error: {str(e)}"}
                step_timings["step6_ittcheck"] = f"{time.time() - step_start:.2f}s"
                working_process_step["step6_ittcheck"] = "Failed"
        else:
            print("\n" + "-"*60)
            print("⏭ STEP 6: ITTCheck Validation - SKIPPED (DISABLED)")
            print("-"*60)
            ittcheck_result = {"error": "Step disabled", "success": False}
            step_timings["step6_ittcheck"] = "0.00s"
            working_process_step["step6_ittcheck"] = "Skipped (Disabled)"

        # STEP 7: AI Parser (Final Fallback) - Only if AI=ON and Step Enabled
        if use_gemini and is_step_enabled("STEP7", step_config_override):
            # Check if Tesseract detected invalid passport image
            if tesseract_result.get("error") == "Not Valid for Passport":
                print("\n" + "-"*60)
                print("🤖 STEP 7: AI Parser (Gemini - Final Fallback)")
                print("-"*60)
                print("  → Cannot get valid image. Give valid image.")
                
                # Clean up user temp folder
                if user_folder_path:
                    from utils import cleanup_user_folder
                    cleanup_user_folder(user_folder_path)
                
                ai_result = {
                    "success": False,
                    "error": "Cannot get valid image. Give valid image."
                }
                step_timings["step7_ai_parser"] = "0.00s"
                working_process_step["step7_ai_parser"] = "Skipped (Invalid Image)"
            else:
                print("\n" + "-"*60)
                print("🤖 STEP 7: AI Parser (Gemini - Final Fallback)")
                print("-"*60)
                
                step_start = time.time()
                
                # Use image_url if available, otherwise use PIL Image
                if image_url:
                    ai_result = gemini_ocr(image_url, is_url=True, user_id=user_id)
                else:
                    ai_result = gemini_ocr(image, is_url=False, user_id=user_id)
            
                step_timings["step7_ai_parser"] = f"{time.time() - step_start:.2f}s"
                working_process_step["step7_ai_parser"] = "AI"
            
            if ai_result.get("success", False):
                print("\n✅ SUCCESS via AI Parser")
                
                # Clean up user temp folder on success
                if user_folder_path:
                    from utils import cleanup_user_folder
                    cleanup_user_folder(user_folder_path)
                
                total_time = time.time() - total_start_time
                
                return {
                    "success": True,
                    "passport_data": {
                        "processVia": "AI",
                        **ai_result.get("passport_data", {})
                    },
                    "mrz_text": ai_result.get("mrz_text", ""),
                    "working_process_step": working_process_step,
                    "step_timings": step_timings,
                    "total_time": f"{total_time:.2f}s",
                    "error": "",
                    "validation_reason": "",
                    "validation_confidence": ""
                }
            
            print(f"  ⚠ AI Parser failed: {ai_result.get('error', 'Unknown error')}")
        else:
            if not use_gemini:
                print("\n⏭ STEP 7: AI Parser - SKIPPED (AI=OFF)")
                working_process_step["step7_ai_parser"] = "Skipped (AI=OFF)"
            elif not is_step_enabled("STEP7", step_config_override):
                print("\n⏭ STEP 7: AI Parser - SKIPPED (DISABLED)")
                working_process_step["step7_ai_parser"] = "Skipped (Disabled)"
            else:
                working_process_step["step7_ai_parser"] = "Skipped"
        
        # All methods failed
        print("\n" + "="*60)
        print("❌ ALL VALIDATION METHODS FAILED")
        print("="*60)
        
        # Clean up user temp folder on failure
        if user_folder_path:
            from utils import cleanup_user_folder
            cleanup_user_folder(user_folder_path)
        
        total_time = time.time() - total_start_time
        
        # Collect all errors
        errors = []
        if fastmrz_result.get("error"):
            errors.append(f"FastMRZ: {fastmrz_result['error']}")
        if passporteye_result.get("error"):
            errors.append(f"PassportEye: {passporteye_result['error']}")
        if easyocr_result.get("error"):
            errors.append(f"EasyOCR: {easyocr_result['error']}")
        if tesseract_result.get("error"):
            errors.append(f"Tesseract: {tesseract_result['error']}")
        if use_gemini:
            if ai_result.get("error"):
                errors.append(f"AI: {ai_result['error']}")
        
        error_message = " | ".join(errors) if errors else "All validation methods failed"
        
        return {
            "success": False,
            "passport_data": {},
            "mrz_text": "",
            "working_process_step": working_process_step,
            "step_timings": step_timings,
            "total_time": f"{total_time:.2f}s",
            "error": error_message,
            "validation_reason": "All extraction methods failed",
            "validation_confidence": "0%"
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n❌ CRITICAL ERROR: {e}")
        print(f"Traceback:\n{error_details}")
        
        # Clean up user temp folder on critical error
        if 'user_folder_path' in locals() and user_folder_path:
            from utils import cleanup_user_folder
            cleanup_user_folder(user_folder_path)
        
        total_time = time.time() - total_start_time
        
        return {
            "success": False,
            "passport_data": {},
            "mrz_text": "",
            "working_process_step": working_process_step,
            "step_timings": step_timings,
            "total_time": f"{total_time:.2f}s",
            "error": f"System error: {str(e)}",
            "validation_reason": "",
            "validation_confidence": ""
        }
