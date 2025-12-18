#!/usr/bin/env python3
"""
Update server configuration for existing DOCscan deployment
Adds Tesseract process management without disrupting service
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command safely"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        return None

def backup_file(filepath):
    """Create backup of existing file"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup"
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backed up {filepath} to {backup_path}")
        return True
    return False

def update_tesseract_config():
    """Update Tesseract configuration in existing files"""
    
    print("🔧 Updating Tesseract configuration...")
    
    # Check if we're in the right directory
    if not os.path.exists('/home/Docscan'):
        print("❌ Not running on DOCscan server (/home/Docscan not found)")
        return False
    
    os.chdir('/home/Docscan')
    
    # 1. Update tesseractOCR.py if it exists
    tesseract_file = 'tesseractOCR.py'
    if os.path.exists(tesseract_file):
        backup_file(tesseract_file)
        
        # Add process management imports
        with open(tesseract_file, 'r') as f:
            content = f.read()
        
        # Check if already updated
        if 'cleanup_tesseract_processes' not in content:
            # Add imports
            import_section = """import psutil
import signal
import subprocess"""
            
            if 'import psutil' not in content:
                content = content.replace('import pytesseract', f'import pytesseract\n{import_section}')
            
            # Add cleanup function
            cleanup_function = '''
def cleanup_tesseract_processes():
    """Clean up any stuck Tesseract processes"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'create_time']):
            try:
                if 'tesseract' in proc.info['name'].lower():
                    # Kill processes older than 30 seconds
                    if psutil.time.time() - proc.info['create_time'] > 30:
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

def safe_tesseract_call(image, config, timeout=10):
    """Safe Tesseract call with cleanup"""
    try:
        cleanup_tesseract_processes()
        return pytesseract.image_to_string(image, config=config, timeout=timeout)
    except Exception as e:
        cleanup_tesseract_processes()
        raise e
'''
            
            # Insert after imports
            lines = content.split('\n')
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.startswith('def ') or line.startswith('class '):
                    insert_pos = i
                    break
            
            lines.insert(insert_pos, cleanup_function)
            content = '\n'.join(lines)
            
            # Replace pytesseract calls
            content = content.replace(
                'pytesseract.image_to_string(processed_img, config=config)',
                'safe_tesseract_call(processed_img, config, timeout=5)'
            )
            content = content.replace(
                'pytesseract.image_to_string(sharpened, config=\'--psm 6\')',
                'safe_tesseract_call(sharpened, \'--psm 6\', timeout=5)'
            )
            
            with open(tesseract_file, 'w') as f:
                f.write(content)
            
            print("✅ Updated tesseractOCR.py with process management")
        else:
            print("✅ tesseractOCR.py already has process management")
    
    # 2. Create process monitor
    monitor_script = '''#!/bin/bash
# DOCscan Tesseract Monitor
MAX_PROCESSES=8
for pid in $(pgrep tesseract); do
    age=$(ps -o etimes= -p $pid 2>/dev/null | tr -d ' ')
    if [ "$age" -gt 30 ]; then
        kill -9 $pid 2>/dev/null
    fi
done

current=$(pgrep tesseract | wc -l)
if [ $current -gt $MAX_PROCESSES ]; then
    pkill -9 -f tesseract
fi
'''
    
    with open('tesseract_monitor.sh', 'w') as f:
        f.write(monitor_script)
    os.chmod('tesseract_monitor.sh', 0o755)
    
    # 3. Add to crontab
    cron_job = "*/2 * * * * /home/Docscan/tesseract_monitor.sh >> /home/Docscan/log/tesseract-monitor.log 2>&1"
    
    # Check if cron job already exists
    existing_cron = run_command("crontab -l 2>/dev/null", check=False) or ""
    if "tesseract_monitor.sh" not in existing_cron:
        run_command(f'(crontab -l 2>/dev/null; echo "{cron_job}") | crontab -')
        print("✅ Added Tesseract monitoring to crontab")
    else:
        print("✅ Tesseract monitoring already in crontab")
    
    # 4. Create emergency cleanup script
    emergency_script = '''#!/bin/bash
echo "🚨 Emergency Tesseract cleanup..."
pkill -9 -f tesseract 2>/dev/null || true
sleep 2
echo "Remaining processes: $(pgrep tesseract | wc -l)"
'''
    
    with open('emergency_tesseract_cleanup.sh', 'w') as f:
        f.write(emergency_script)
    os.chmod('emergency_tesseract_cleanup.sh', 0o755)
    
    print("✅ Created emergency cleanup script")
    
    return True

def show_status():
    """Show current system status"""
    print("\n📊 Current Status:")
    print("=" * 30)
    
    # Service status
    status = run_command("systemctl is-active docscan.service", check=False)
    print(f"DOCscan service: {status}")
    
    # Tesseract processes
    tesseract_count = run_command("pgrep tesseract | wc -l", check=False) or "0"
    print(f"Tesseract processes: {tesseract_count}")
    
    # Memory usage
    memory = run_command("free -h | grep Mem | awk '{print $3\"/\"$2}'", check=False)
    print(f"Memory usage: {memory}")
    
    # Load average
    load = run_command("uptime | awk -F'load average:' '{print $2}'", check=False)
    print(f"Load average:{load}")

def main():
    print("🔧 DOCscan Server Configuration Update")
    print("=" * 40)
    
    if os.geteuid() != 0:
        print("⚠️  Running as non-root user. Some operations may require sudo.")
    
    # Show current status
    show_status()
    
    # Update configuration
    if update_tesseract_config():
        print("\n✅ Configuration updated successfully!")
        
        print("\n📋 What was done:")
        print("• Added Tesseract process cleanup functions")
        print("• Created monitoring script (tesseract_monitor.sh)")
        print("• Added cron job for automatic cleanup every 2 minutes")
        print("• Created emergency cleanup script")
        
        print("\n🛠️  Available commands:")
        print("• Monitor: ./tesseract_monitor.sh")
        print("• Emergency cleanup: ./emergency_tesseract_cleanup.sh")
        print("• View logs: tail -f log/tesseract-monitor.log")
        
        print("\n⚠️  Recommendations:")
        print("• Consider reducing workers from 4 to 2 in systemd service")
        print("• Monitor /home/Docscan/log/tesseract-monitor.log")
        print("• Run emergency cleanup if processes accumulate")
        
    else:
        print("\n❌ Configuration update failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())