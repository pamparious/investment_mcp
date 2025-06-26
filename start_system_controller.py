#!/usr/bin/env python3
"""
Start the Investment MCP System Controller
This provides the backend API for the dashboard's system control features
"""

import sys
import subprocess
from pathlib import Path

def main():
    # Get the path to the system controller
    controller_path = Path(__file__).parent / "backend" / "system_controller" / "main.py"
    
    if not controller_path.exists():
        print(f"Error: System controller not found at {controller_path}")
        sys.exit(1)
    
    print("Starting Investment MCP System Controller on port 8001...")
    print("This will allow the dashboard to control backend services.")
    print("Press Ctrl+C to stop.")
    
    try:
        # Start the system controller
        subprocess.run([sys.executable, str(controller_path)], check=True)
    except KeyboardInterrupt:
        print("\nSystem controller stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error running system controller: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()