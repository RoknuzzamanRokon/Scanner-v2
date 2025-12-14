"""
Function Handler Switch System
Allows enabling/disabling specific steps in the passport scanning process
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class StepController:
    """
    Controls which steps are enabled/disabled in the scanning process
    """
    
    def __init__(self):
        """Initialize step controller with default settings"""
        self.steps = {
            "STEP1": True,   # FastMRZ
            "STEP2": True,   # PassportEye  
            "STEP3": True,   # EasyOCR
            "STEP4": True,   # Tesseract
            "STEP5": True,   # Validation Checker
            "STEP6": True,   # AI Parser
        }
        
        # Load settings from environment variables
        self._load_from_env()
        
        # Load settings from config file if exists
        self._load_from_config()
    
    def _load_from_env(self):
        """Load step settings from environment variables"""
        for step in self.steps.keys():
            env_value = os.getenv(f"{step}_ENABLED", "").lower()
            if env_value in ["off", "false", "0", "disabled"]:
                self.steps[step] = False
                print(f"🔴 {step} disabled via environment variable")
            elif env_value in ["on", "true", "1", "enabled"]:
                self.steps[step] = True
                print(f"🟢 {step} enabled via environment variable")
    
    def _load_from_config(self):
        """Load step settings from config file"""
        try:
            config_file = "step_config.txt"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            step, value = line.split('=', 1)
                            step = step.strip().upper()
                            value = value.strip().lower()
                            
                            if step in self.steps:
                                if value in ["off", "false", "0", "disabled"]:
                                    self.steps[step] = False
                                    print(f"🔴 {step} disabled via config file")
                                elif value in ["on", "true", "1", "enabled"]:
                                    self.steps[step] = True
                                    print(f"🟢 {step} enabled via config file")
        except Exception as e:
            print(f"⚠️ Error loading config file: {e}")
    
    def is_enabled(self, step: str) -> bool:
        """
        Check if a step is enabled
        
        Args:
            step: Step name (e.g., "STEP1", "STEP2", etc.)
            
        Returns:
            True if step is enabled, False otherwise
        """
        step = step.upper()
        return self.steps.get(step, True)  # Default to True if step not found
    
    def enable_step(self, step: str):
        """Enable a specific step"""
        step = step.upper()
        if step in self.steps:
            self.steps[step] = True
            print(f"🟢 {step} enabled")
        else:
            print(f"❌ Unknown step: {step}")
    
    def disable_step(self, step: str):
        """Disable a specific step"""
        step = step.upper()
        if step in self.steps:
            self.steps[step] = False
            print(f"🔴 {step} disabled")
        else:
            print(f"❌ Unknown step: {step}")
    
    def get_status(self) -> Dict[str, bool]:
        """Get current status of all steps"""
        return self.steps.copy()
    
    def print_status(self):
        """Print current status of all steps"""
        print("\n" + "="*50)
        print("📊 STEP CONTROLLER STATUS")
        print("="*50)
        
        step_names = {
            "STEP1": "FastMRZ Fallback",
            "STEP2": "PassportEye Fallback", 
            "STEP3": "EasyOCR Fallback",
            "STEP4": "Tesseract OCR Fallback",
            "STEP5": "Passport Validation Checker",
            "STEP6": "AI Parser (Gemini)"
        }
        
        for step, enabled in self.steps.items():
            status = "🟢 ENABLED " if enabled else "🔴 DISABLED"
            name = step_names.get(step, "Unknown Step")
            print(f"{step}: {status} - {name}")
        
        print("="*50)
    
    def save_config(self):
        """Save current settings to config file"""
        try:
            with open("step_config.txt", 'w') as f:
                f.write("# Step Controller Configuration\n")
                f.write("# Set to 'on' or 'off' to enable/disable steps\n")
                f.write("# STEP1 = FastMRZ Fallback\n")
                f.write("# STEP2 = PassportEye Fallback\n")
                f.write("# STEP3 = EasyOCR Fallback\n")
                f.write("# STEP4 = Tesseract OCR Fallback\n")
                f.write("# STEP5 = Passport Validation Checker\n")
                f.write("# STEP6 = AI Parser (Gemini)\n\n")
                
                for step, enabled in self.steps.items():
                    value = "on" if enabled else "off"
                    f.write(f"{step}={value}\n")
            
            print("✅ Configuration saved to step_config.txt")
        except Exception as e:
            print(f"❌ Error saving config: {e}")


# Global step controller instance
step_controller = StepController()


def is_step_enabled(step: str) -> bool:
    """
    Check if a step is enabled (convenience function)
    
    Args:
        step: Step name (e.g., "STEP1", "STEP2", etc.)
        
    Returns:
        True if step is enabled, False otherwise
    """
    return step_controller.is_enabled(step)


def get_step_status() -> Dict[str, bool]:
    """Get current status of all steps (convenience function)"""
    return step_controller.get_status()


def print_step_status():
    """Print current status of all steps (convenience function)"""
    step_controller.print_status()


def enable_step(step: str):
    """Enable a specific step (convenience function)"""
    step_controller.enable_step(step)


def disable_step(step: str):
    """Disable a specific step (convenience function)"""
    step_controller.disable_step(step)


def save_step_config():
    """Save current step configuration (convenience function)"""
    step_controller.save_config()


# Example usage and testing
if __name__ == "__main__":
    print("🔧 Step Controller Test")
    
    # Print initial status
    print_step_status()
    
    # Test disabling steps
    print("\n🔴 Testing step disabling:")
    disable_step("STEP3")  # Disable EasyOCR
    disable_step("STEP4")  # Disable Tesseract
    
    # Print updated status
    print_step_status()
    
    # Test enabling steps
    print("\n🟢 Testing step enabling:")
    enable_step("STEP3")   # Re-enable EasyOCR
    
    # Print final status
    print_step_status()
    
    # Test checking individual steps
    print("\n🔍 Testing individual step checks:")
    print(f"STEP1 enabled: {is_step_enabled('STEP1')}")
    print(f"STEP3 enabled: {is_step_enabled('STEP3')}")
    print(f"STEP4 enabled: {is_step_enabled('STEP4')}")
    
    # Save configuration
    save_step_config()