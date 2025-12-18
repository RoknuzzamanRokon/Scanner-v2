#!/bin/bash

# Quick fix for running DOCscan server
# Addresses Tesseract process accumulation

echo "🔧 Fixing Tesseract issues on running DOCscan server"
echo "=================================================="

# 1. Kill existing stuck Tesseract processes
echo "🧹 Cleaning up stuck Tesseract processes..."
pkill -f tesseract 2>/dev/null || true
sleep 2

# Count remaining processes
REMAINING=$(pgrep tesseract | wc -l)
echo "   Remaining Tesseract processes: $REMAINING"

# 2. Create process monitor script
echo "📊 Creating process monitor..."
sudo tee /usr/local/bin/docscan-tesseract-monitor.sh << 'EOF'
#!/bin/bash
# DOCscan Tesseract Process Monitor

LOG_FILE="/home/Docscan/log/tesseract-monitor.log"
MAX_PROCESSES=8  # 4 workers * 2 processes each
MAX_AGE=30       # Kill processes older than 30 seconds

# Function to log with timestamp
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Count current processes
CURRENT_COUNT=$(pgrep tesseract | wc -l)

if [ $CURRENT_COUNT -gt $MAX_PROCESSES ]; then
    log_msg "WARNING: Too many Tesseract processes ($CURRENT_COUNT > $MAX_PROCESSES)"
    
    # Kill old processes
    for pid in $(pgrep tesseract); do
        AGE=$(ps -o etimes= -p $pid 2>/dev/null | tr -d ' ')
        if [ ! -z "$AGE" ] && [ "$AGE" -gt $MAX_AGE ]; then
            log_msg "Killing old Tesseract process $pid (age: ${AGE}s)"
            kill -9 $pid 2>/dev/null
        fi
    done
    
    # Final count
    NEW_COUNT=$(pgrep tesseract | wc -l)
    log_msg "Cleanup complete: $CURRENT_COUNT -> $NEW_COUNT processes"
fi
EOF

sudo chmod +x /usr/local/bin/docscan-tesseract-monitor.sh

# 3. Create systemd timer for monitoring
echo "⏰ Setting up monitoring timer..."
sudo tee /etc/systemd/system/docscan-tesseract-cleanup.service << 'EOF'
[Unit]
Description=DOCscan Tesseract Process Cleanup
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/docscan-tesseract-monitor.sh
User=root
EOF

sudo tee /etc/systemd/system/docscan-tesseract-cleanup.timer << 'EOF'
[Unit]
Description=Run DOCscan Tesseract cleanup every 2 minutes
Requires=docscan-tesseract-cleanup.service

[Timer]
OnCalendar=*:0/2
Persistent=true

[Install]
WantedBy=timers.target
EOF

# 4. Enable and start the timer
sudo systemctl daemon-reload
sudo systemctl enable docscan-tesseract-cleanup.timer
sudo systemctl start docscan-tesseract-cleanup.timer

# 5. Create emergency cleanup script
echo "🚨 Creating emergency cleanup script..."
tee /home/Docscan/emergency_cleanup.sh << 'EOF'
#!/bin/bash
# Emergency Tesseract cleanup for DOCscan

echo "🚨 EMERGENCY TESSERACT CLEANUP"
echo "=============================="

# Show current processes
echo "Current Tesseract processes:"
ps aux | grep tesseract | grep -v grep

# Kill all Tesseract processes
echo ""
echo "Killing all Tesseract processes..."
pkill -9 -f tesseract 2>/dev/null || true

# Wait and check
sleep 3
REMAINING=$(pgrep tesseract | wc -l)
echo "Remaining processes: $REMAINING"

if [ $REMAINING -eq 0 ]; then
    echo "✅ All Tesseract processes cleaned up"
else
    echo "⚠️  Some processes may still be running"
    ps aux | grep tesseract | grep -v grep
fi

# Restart DOCscan service
echo ""
echo "🔄 Restarting DOCscan service..."
sudo systemctl restart docscan.service
echo "✅ Service restarted"
EOF

chmod +x /home/Docscan/emergency_cleanup.sh

# 6. Update the existing service to include resource limits
echo "⚙️  Updating DOCscan service with resource limits..."
sudo tee /etc/systemd/system/docscan.service << 'EOF'
[Unit]
Description=DOCscan FastAPI App (Uvicorn)
After=network.target

[Service]
# run as root, because you are using root user
User=root
Group=root

# your project directory
WorkingDirectory=/home/Docscan

# load env file
EnvironmentFile=/home/Docscan/.env

# Resource limits to prevent process accumulation
LimitNOFILE=2048
LimitNPROC=100
CPUQuota=200%
MemoryLimit=2G

# start using YOUR venv with reduced workers to prevent Tesseract overload
ExecStart=/home/Docscan/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2

# auto restart
Restart=always
RestartSec=5

# logging
StandardOutput=append:/home/Docscan/log/app.log
StandardError=append:/home/Docscan/log/error.log

[Install]
WantedBy=multi-user.target
EOF

# 7. Create monitoring dashboard script
echo "📊 Creating monitoring dashboard..."
tee /home/Docscan/monitor_dashboard.sh << 'EOF'
#!/bin/bash
# DOCscan Monitoring Dashboard

clear
echo "🔍 DOCscan Server Monitor"
echo "========================"
echo ""

# Service status
echo "📋 Service Status:"
systemctl is-active docscan.service
echo ""

# Tesseract processes
TESSERACT_COUNT=$(pgrep tesseract | wc -l)
echo "🔧 Tesseract Processes: $TESSERACT_COUNT"
if [ $TESSERACT_COUNT -gt 0 ]; then
    echo "   PIDs: $(pgrep tesseract | tr '\n' ' ')"
    echo "   Details:"
    ps aux | grep tesseract | grep -v grep | awk '{print "   PID:"$2" CPU:"$3"% MEM:"$4"% TIME:"$10}'
fi
echo ""

# Memory usage
echo "💾 Memory Usage:"
free -h | grep Mem | awk '{print "   Used: "$3" / "$2" ("$3/$2*100"%)"}'
echo ""

# CPU load
echo "⚡ CPU Load:"
uptime | awk -F'load average:' '{print "   Load:"$2}'
echo ""

# Recent logs
echo "📝 Recent Errors (last 10 lines):"
tail -10 /home/Docscan/log/error.log 2>/dev/null || echo "   No error log found"
echo ""

# Cleanup timer status
echo "⏰ Cleanup Timer:"
systemctl is-active docscan-tesseract-cleanup.timer
echo ""

echo "🛠️  Commands:"
echo "   Emergency cleanup: ./emergency_cleanup.sh"
echo "   Restart service: sudo systemctl restart docscan.service"
echo "   View logs: tail -f log/app.log"
echo "   Kill Tesseract: pkill -f tesseract"
EOF

chmod +x /home/Docscan/monitor_dashboard.sh

# 8. Reload and restart with new configuration
echo "🔄 Applying changes..."
sudo systemctl daemon-reload

# Ask user if they want to restart now
echo ""
echo "⚠️  To apply resource limits, the service needs to be restarted."
echo "   Current workers: 4 -> New workers: 2 (to reduce Tesseract load)"
echo ""
read -p "Restart DOCscan service now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo systemctl restart docscan.service
    echo "✅ Service restarted with new configuration"
else
    echo "⏸️  Service not restarted. Run 'sudo systemctl restart docscan.service' when ready."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 Monitoring:"
echo "   Dashboard: ./monitor_dashboard.sh"
echo "   Emergency cleanup: ./emergency_cleanup.sh"
echo "   Timer status: systemctl status docscan-tesseract-cleanup.timer"
echo ""
echo "📁 Log files:"
echo "   App logs: /home/Docscan/log/app.log"
echo "   Error logs: /home/Docscan/log/error.log"
echo "   Tesseract monitor: /home/Docscan/log/tesseract-monitor.log"