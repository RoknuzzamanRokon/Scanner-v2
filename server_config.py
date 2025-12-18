"""
Server configuration for production deployment
Handles Tesseract process management and resource limits
"""
import os
import psutil
import signal
import subprocess
from typing import Optional

class TesseractServerManager:
    """Manage Tesseract processes on server"""
    
    def __init__(self):
        self.max_processes = 3  # Maximum concurrent Tesseract processes
        self.process_timeout = 15  # Maximum time per process (seconds)
        self.cleanup_interval = 60  # Cleanup interval (seconds)
    
    def get_tesseract_processes(self):
        """Get all running Tesseract processes"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'create_time', 'cpu_percent']):
                if 'tesseract' in proc.info['name'].lower():
                    processes.append(proc)
        except Exception:
            pass
        return processes
    
    def cleanup_old_processes(self):
        """Clean up old/stuck Tesseract processes"""
        cleaned = 0
        try:
            current_time = psutil.time.time()
            for proc in self.get_tesseract_processes():
                try:
                    # Kill processes older than timeout
                    if current_time - proc.info['create_time'] > self.process_timeout:
                        proc.kill()
                        cleaned += 1
                        print(f"Killed stuck Tesseract process (PID: {proc.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"Error during cleanup: {e}")
        return cleaned
    
    def can_start_new_process(self):
        """Check if we can start a new Tesseract process"""
        current_processes = len(self.get_tesseract_processes())
        return current_processes < self.max_processes
    
    def wait_for_slot(self, max_wait=30):
        """Wait for an available process slot"""
        import time
        waited = 0
        while not self.can_start_new_process() and waited < max_wait:
            time.sleep(1)
            waited += 1
            if waited % 5 == 0:  # Cleanup every 5 seconds
                self.cleanup_old_processes()
        return self.can_start_new_process()

# Global instance
tesseract_manager = TesseractServerManager()

def server_safe_tesseract(image, config, timeout=10):
    """
    Server-safe Tesseract execution with process management
    """
    import pytesseract
    
    # Wait for available slot
    if not tesseract_manager.wait_for_slot():
        raise Exception("Tesseract server busy - too many concurrent processes")
    
    try:
        # Run with timeout
        result = pytesseract.image_to_string(image, config=config, timeout=timeout)
        return result
    except Exception as e:
        # Cleanup on error
        tesseract_manager.cleanup_old_processes()
        raise e

def setup_server_limits():
    """Setup server resource limits"""
    try:
        # Set process limits
        import resource
        
        # Limit memory usage (1GB)
        resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024, 1024*1024*1024))
        
        # Limit CPU time (30 seconds per process)
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        
        print("Server resource limits configured")
    except Exception as e:
        print(f"Could not set resource limits: {e}")

def create_systemd_cleanup_service():
    """Create systemd service for periodic cleanup"""
    service_content = """[Unit]
Description=Tesseract Process Cleanup
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 -c "
import psutil
import time
for proc in psutil.process_iter(['pid', 'name', 'create_time']):
    try:
        if 'tesseract' in proc.info['name'].lower():
            if time.time() - proc.info['create_time'] > 60:
                proc.kill()
                print(f'Killed stuck tesseract process {proc.info[\"pid\"]}')
    except:
        pass
"

[Install]
WantedBy=multi-user.target
"""
    
    timer_content = """[Unit]
Description=Run Tesseract cleanup every 5 minutes
Requires=tesseract-cleanup.service

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
"""
    
    try:
        with open('/tmp/tesseract-cleanup.service', 'w') as f:
            f.write(service_content)
        
        with open('/tmp/tesseract-cleanup.timer', 'w') as f:
            f.write(timer_content)
        
        print("Systemd service files created in /tmp/")
        print("To install:")
        print("sudo cp /tmp/tesseract-cleanup.* /etc/systemd/system/")
        print("sudo systemctl enable tesseract-cleanup.timer")
        print("sudo systemctl start tesseract-cleanup.timer")
        
    except Exception as e:
        print(f"Could not create systemd files: {e}")

if __name__ == "__main__":
    print("Tesseract Server Manager")
    print("=" * 30)
    
    # Show current status
    processes = tesseract_manager.get_tesseract_processes()
    print(f"Current Tesseract processes: {len(processes)}")
    
    for proc in processes:
        try:
            print(f"  PID: {proc.info['pid']}, Age: {psutil.time.time() - proc.info['create_time']:.1f}s")
        except:
            pass
    
    # Cleanup old processes
    cleaned = tesseract_manager.cleanup_old_processes()
    print(f"Cleaned up {cleaned} old processes")
    
    # Create systemd service
    create_systemd_cleanup_service()