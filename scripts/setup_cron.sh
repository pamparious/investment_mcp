#!/bin/bash
# Setup script for daily data pipeline cron job

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DAILY_SCRIPT="$PROJECT_DIR/scripts/daily_data_update.py"

echo "Setting up daily data pipeline cron job..."
echo "Project directory: $PROJECT_DIR"
echo "Daily script: $DAILY_SCRIPT"

# Create cron job entry
CRON_ENTRY="0 6 * * 1-5 cd $PROJECT_DIR && /usr/bin/python3 $DAILY_SCRIPT >> logs/cron.log 2>&1"

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "✅ Cron job added successfully!"
echo "📅 Schedule: Every weekday at 6:00 AM"
echo "📝 Logs will be written to: $PROJECT_DIR/logs/cron.log"
echo ""
echo "To view current cron jobs: crontab -l"
echo "To remove this cron job: crontab -e"
echo ""
echo "Manual execution:"
echo "  cd $PROJECT_DIR"
echo "  python3 scripts/daily_data_update.py --help"