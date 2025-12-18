#!/bin/bash

# Server deployment script for passport scanner
# Handles Tesseract installation and process management

echo "🚀 Setting up Passport Scanner Server"
echo "=================================="

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng python3-pip htop

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install -r requirements.txt

# Set up Tesseract limits
echo "⚙️  Configuring Tesseract limits..."

# Create limits configuration
sudo tee /etc/security/limits.d/tesseract.conf << EOF
# Tesseract process limits
* soft nproc 10
* hard nproc 20
* soft nofile 1024
* hard nofile 2048
EOF

# Create systemd service for cleanup
echo "🧹 Setting up process cleanup service..."
python3 server_config.py

# Create monitoring script
sudo tee /usr/local/bin/monitor-tesseract.sh << 'EOF'
#!/bin/bash
# Monitor Tesseract processes

echo "Tesseract Process Monitor"
echo "========================"
echo "Current processes:"
ps aux | grep tesseract | grep -v grep

echo ""
echo "Process count: $(pgrep tesseract | wc -l)"

# Kill processes older than 60 seconds
for pid in $(pgrep tesseract); do
    age=$(ps -o etimes= -p $pid 2>/dev/null | tr -d ' ')
    if [ "$age" -gt 60 ]; then
        echo "Killing old process $pid (age: ${age}s)"
        kill -9 $pid 2>/dev/null
    fi
done
EOF

sudo chmod +x /usr/local/bin/monitor-tesseract.sh

# Create cron job for monitoring
echo "⏰ Setting up monitoring cron job..."
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/monitor-tesseract.sh >> /var/log/tesseract-monitor.log 2>&1") | crontab -

# Set up log rotation
sudo tee /etc/logrotate.d/tesseract-monitor << EOF
/var/log/tesseract-monitor.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF

# Create startup script
tee start_server.sh << 'EOF'
#!/bin/bash
# Start the passport scanner server

echo "🔧 Pre-flight checks..."

# Clean up any existing processes
pkill -f tesseract 2>/dev/null || true

# Check Tesseract installation
if ! command -v tesseract &> /dev/null; then
    echo "❌ Tesseract not found!"
    exit 1
fi

echo "✅ Tesseract found: $(tesseract --version | head -1)"

# Set resource limits
ulimit -n 1024  # File descriptors
ulimit -u 50    # Processes
ulimit -t 30    # CPU time (seconds)

echo "🚀 Starting server..."
python3 app.py
EOF

chmod +x start_server.sh

# Create environment file template
tee .env.server << 'EOF'
# Server Environment Configuration
TESSERACT_CMD=/usr/bin/tesseract
GEMINI_API_KEY=your_gemini_api_key_here

# Server limits
MAX_TESSERACT_PROCESSES=3
TESSERACT_TIMEOUT=15
PROCESS_CLEANUP_INTERVAL=60

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/passport-scanner.log
EOF

echo ""
echo "✅ Server setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy your .env file: cp .env.server .env"
echo "2. Add your Gemini API key to .env"
echo "3. Start the server: ./start_server.sh"
echo ""
echo "🔍 Monitoring:"
echo "- Check processes: /usr/local/bin/monitor-tesseract.sh"
echo "- View logs: tail -f /var/log/tesseract-monitor.log"
echo "- System monitor: htop"
echo ""
echo "🛠️  Troubleshooting:"
echo "- Kill all Tesseract: pkill -f tesseract"
echo "- Check limits: ulimit -a"
echo "- Monitor resources: htop"