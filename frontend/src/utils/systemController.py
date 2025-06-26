#!/usr/bin/env python3
"""
System Controller for Investment MCP
Provides HTTP API to control backend services, MCP server, and system operations
"""

import os
import sys
import subprocess
import json
import signal
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import psutil

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

app = FastAPI(title="Investment MCP System Controller", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://172.29.41.222:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class ProcessInfo:
    pid: Optional[int]
    status: str
    uptime: Optional[str]
    port: int

class SystemStatus(BaseModel):
    backend_api: Dict[str, Any]
    mcp_server: Dict[str, Any]
    database: Dict[str, Any]
    data_collectors: Dict[str, Any]

class ServiceControl:
    def __init__(self):
        self.project_root = project_root
        self.processes: Dict[str, ProcessInfo] = {
            'backend': ProcessInfo(None, 'stopped', None, 8000),
            'mcp': ProcessInfo(None, 'stopped', None, 3001)
        }
    
    def get_process_info(self, service: str) -> ProcessInfo:
        """Get current process information for a service"""
        if service not in self.processes:
            return ProcessInfo(None, 'unknown', None, 0)
        
        proc_info = self.processes[service]
        
        # Check if process is actually running
        if proc_info.pid:
            try:
                process = psutil.Process(proc_info.pid)
                if process.is_running():
                    uptime = time.time() - process.create_time()
                    proc_info.uptime = f"{int(uptime//3600)}h {int((uptime%3600)//60)}m"
                    proc_info.status = 'running'
                else:
                    proc_info.pid = None
                    proc_info.status = 'stopped'
                    proc_info.uptime = None
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                proc_info.pid = None
                proc_info.status = 'stopped'
                proc_info.uptime = None
        
        return proc_info
    
    def start_backend_api(self) -> Dict[str, Any]:
        """Start the FastAPI backend server"""
        try:
            # Check if already running
            proc_info = self.get_process_info('backend')
            if proc_info.status == 'running':
                return {"success": False, "message": "Backend API is already running"}
            
            # Start the backend server
            cmd = [
                sys.executable, "-m", "uvicorn",
                "investment_mcp_api.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "PYTHONPATH": str(self.project_root)}
            )
            
            # Wait a moment to see if it starts successfully
            time.sleep(2)
            if process.poll() is None:  # Still running
                self.processes['backend'].pid = process.pid
                self.processes['backend'].status = 'running'
                return {"success": True, "message": f"Backend API started with PID {process.pid}"}
            else:
                stdout, stderr = process.communicate()
                return {"success": False, "message": f"Failed to start backend: {stderr.decode()}"}
                
        except Exception as e:
            return {"success": False, "message": f"Error starting backend: {str(e)}"}
    
    def stop_backend_api(self) -> Dict[str, Any]:
        """Stop the FastAPI backend server"""
        try:
            proc_info = self.get_process_info('backend')
            if proc_info.status != 'running' or not proc_info.pid:
                return {"success": False, "message": "Backend API is not running"}
            
            # Terminate the process
            try:
                process = psutil.Process(proc_info.pid)
                process.terminate()
                process.wait(timeout=10)
                
                self.processes['backend'].pid = None
                self.processes['backend'].status = 'stopped'
                self.processes['backend'].uptime = None
                
                return {"success": True, "message": "Backend API stopped successfully"}
            except psutil.TimeoutExpired:
                process.kill()
                return {"success": True, "message": "Backend API forcefully stopped"}
                
        except Exception as e:
            return {"success": False, "message": f"Error stopping backend: {str(e)}"}
    
    def start_mcp_server(self) -> Dict[str, Any]:
        """Start the MCP server"""
        try:
            proc_info = self.get_process_info('mcp')
            if proc_info.status == 'running':
                return {"success": False, "message": "MCP server is already running"}
            
            # Start MCP server
            cmd = [
                sys.executable, "-m", "mcp",
                "src.mcp_tools.server"
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "PYTHONPATH": str(self.project_root)}
            )
            
            time.sleep(2)
            if process.poll() is None:
                self.processes['mcp'].pid = process.pid
                self.processes['mcp'].status = 'running'
                return {"success": True, "message": f"MCP server started with PID {process.pid}"}
            else:
                stdout, stderr = process.communicate()
                return {"success": False, "message": f"Failed to start MCP server: {stderr.decode()}"}
                
        except Exception as e:
            return {"success": False, "message": f"Error starting MCP server: {str(e)}"}
    
    def stop_mcp_server(self) -> Dict[str, Any]:
        """Stop the MCP server"""
        try:
            proc_info = self.get_process_info('mcp')
            if proc_info.status != 'running' or not proc_info.pid:
                return {"success": False, "message": "MCP server is not running"}
            
            try:
                process = psutil.Process(proc_info.pid)
                process.terminate()
                process.wait(timeout=10)
                
                self.processes['mcp'].pid = None
                self.processes['mcp'].status = 'stopped'
                self.processes['mcp'].uptime = None
                
                return {"success": True, "message": "MCP server stopped successfully"}
            except psutil.TimeoutExpired:
                process.kill()
                return {"success": True, "message": "MCP server forcefully stopped"}
                
        except Exception as e:
            return {"success": False, "message": f"Error stopping MCP server: {str(e)}"}
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        db_path = self.project_root / "investment_data.db"
        
        try:
            if db_path.exists():
                size = db_path.stat().st_size
                size_mb = size / (1024 * 1024)
                
                # Get table information
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                return {
                    "status": "connected",
                    "path": str(db_path),
                    "size": f"{size_mb:.2f} MB",
                    "tables": tables,
                    "last_updated": datetime.fromtimestamp(db_path.stat().st_mtime).isoformat()
                }
            else:
                return {
                    "status": "disconnected",
                    "path": str(db_path),
                    "size": "0 MB",
                    "tables": [],
                }
        except Exception as e:
            return {
                "status": "error",
                "path": str(db_path),
                "error": str(e)
            }

# Global service controller
controller = ServiceControl()

@app.get("/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get comprehensive system status"""
    backend_info = controller.get_process_info('backend')
    mcp_info = controller.get_process_info('mcp')
    db_info = controller.get_database_info()
    
    return SystemStatus(
        backend_api={
            "status": backend_info.status,
            "port": backend_info.port,
            "pid": backend_info.pid,
            "uptime": backend_info.uptime
        },
        mcp_server={
            "status": mcp_info.status,
            "port": mcp_info.port,
            "pid": mcp_info.pid,
            "tools_available": 10 if mcp_info.status == 'running' else 0
        },
        database=db_info,
        data_collectors={
            "status": "idle",
            "last_run": "2024-01-01T00:00:00",
            "collected_records": 1250
        }
    )

@app.post("/system/backend/start")
async def start_backend():
    """Start the backend API server"""
    result = controller.start_backend_api()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/system/backend/stop")
async def stop_backend():
    """Stop the backend API server"""
    result = controller.stop_backend_api()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/system/backend/restart")
async def restart_backend():
    """Restart the backend API server"""
    stop_result = controller.stop_backend_api()
    time.sleep(2)
    start_result = controller.start_backend_api()
    
    if not start_result["success"]:
        raise HTTPException(status_code=400, detail=start_result["message"])
    
    return {"success": True, "message": "Backend API restarted successfully"}

@app.post("/system/mcp/start")
async def start_mcp():
    """Start the MCP server"""
    result = controller.start_mcp_server()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/system/mcp/stop")
async def stop_mcp():
    """Stop the MCP server"""
    result = controller.stop_mcp_server()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/system/mcp/restart")
async def restart_mcp():
    """Restart the MCP server"""
    stop_result = controller.stop_mcp_server()
    time.sleep(2)
    start_result = controller.start_mcp_server()
    
    if not start_result["success"]:
        raise HTTPException(status_code=400, detail=start_result["message"])
    
    return {"success": True, "message": "MCP server restarted successfully"}

@app.get("/system/logs")
async def get_system_logs(component: str = None, level: str = None, limit: int = 100):
    """Get system logs"""
    # Mock logs for now
    logs = [
        {
            "timestamp": datetime.now().isoformat(),
            "level": "info",
            "component": "system",
            "message": "System controller started"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "level": "info",
            "component": "dashboard",
            "message": "Dashboard connected to system controller"
        }
    ]
    
    return {"logs": logs}

if __name__ == "__main__":
    print("Starting Investment MCP System Controller...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")