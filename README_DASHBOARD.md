# Investment MCP Dashboard - Complete Testing & Operations Tool

## Overview
The Investment MCP Dashboard is now a comprehensive testing and operational tool for the entire Investment MCP system. It provides full control over backend services, MCP server management, API testing, and system monitoring.

## Features

### 1. Dashboard Overview
- **Real-time API Health Monitoring** - Shows connection status to backend API
- **System Status** - Overview of all system components
- **Market Sentiment** - Live market analysis with AI insights
- **Economic Indicators** - Swedish economic data monitoring
- **Quick Actions** - Direct access to all tools

### 2. Portfolio Analysis
- **Interactive Risk Assessment** - Risk tolerance and investment horizon selection
- **AI-Powered Recommendations** - Portfolio optimization with Swedish funds
- **Real-time Analysis** - Portfolio metrics and risk assessment
- **Visual Results** - Charts and tables for analysis results

### 3. Fund Explorer
- **12 Swedish Funds** - Complete universe of approved investment funds
- **Advanced Filtering** - Search by category, risk level, and name
- **Detailed Fund Information** - Performance data, fees, and descriptions
- **Fund Comparison** - Side-by-side fund analysis

### 4. Market Data & Visualization
- **Swedish Economic Indicators** - Riksbank and SCB data integration
- **Interactive Charts** - Economic trends and sector performance
- **Market Sentiment Analysis** - AI-driven market insights
- **Housing Market Data** - Swedish real estate trends

### 5. System Control Center ⭐ NEW
- **Backend API Control** - Start, stop, restart FastAPI server
- **MCP Server Management** - Control MCP server lifecycle
- **Database Operations** - Backup, reset, and monitor database
- **Data Collection Jobs** - Trigger and monitor data collection
- **System Logs** - Real-time log monitoring with filtering
- **Process Monitoring** - Live status of all system processes

### 6. API Testing Tool ⭐ NEW
- **Interactive API Testing** - Test all Investment MCP API endpoints
- **Request Builder** - Visual request construction with parameters
- **Response Analysis** - Detailed response inspection
- **Test History** - Track all API test results
- **API Documentation** - Built-in endpoint documentation

## Getting Started

### Prerequisites
```bash
# Install Python dependencies
pip install fastapi uvicorn psutil

# Install Node.js dependencies (already done)
cd frontend
npm install
```

### Starting the Complete System

1. **Start the Dashboard** (Already running)
   ```bash
   cd frontend
   npm start
   # Dashboard available at: http://localhost:3000
   ```

2. **Start the System Controller** (For advanced features)
   ```bash
   cd /home/ndreasils/investment_mcp
   python start_system_controller.py
   # System controller available at: http://localhost:8001
   ```

3. **Access the Dashboard**
   - Open `http://localhost:3000` in your browser
   - Navigate between pages using the sidebar
   - Use System Control to manage backend services

## How to Use for Testing

### Testing the Investment MCP System

1. **Check System Status**
   - Go to Dashboard → View API connection status
   - Go to System Control → Monitor all system components

2. **Start Backend Services**
   - Go to System Control → Backend API → Click "Start"
   - Go to System Control → MCP Server → Click "Start"
   - Monitor process status in real-time

3. **Test API Endpoints**
   - Go to API Tester → Select endpoint
   - Configure parameters and request body
   - Execute tests and view responses
   - Track test history

4. **Collect Data**
   - Go to System Control → Data Collection → Start Collection
   - Monitor job progress in real-time
   - View collection logs

5. **Test Portfolio Analysis**
   - Go to Portfolio Analysis → Configure investment profile
   - Submit analysis request
   - View AI-powered recommendations

6. **Monitor System Health**
   - Go to System Control → System Logs → Filter by component
   - Monitor database size and operations
   - Track system performance

### Advanced Operations

1. **Database Management**
   - Backup database through System Control
   - Reset database for testing
   - Monitor database size and tables

2. **MCP Tool Testing**
   - Go to System Control → MCP Tools → Test individual tools
   - View tool usage statistics
   - Monitor tool availability

3. **Data Collection Control**
   - Start specific data sources (YFinance, Riksbank, SCB)
   - Cancel running jobs
   - Monitor collection progress

## System Architecture

```
Frontend Dashboard (Port 3000)
├── System Control Interface
├── API Testing Tool  
├── Portfolio Analysis
├── Fund Explorer
└── Market Data Visualization

System Controller (Port 8001)
├── Backend API Control
├── MCP Server Management
├── Database Operations
└── Process Monitoring

Investment MCP Backend (Port 8000)
├── Portfolio Analysis API
├── Fund Data API
├── Economic Data API
└── AI Recommendation Engine

MCP Server (Port 3001)
├── Data Collection Tools
├── Analysis Tools
├── Portfolio Optimization
└── Market Intelligence
```

## Current Status

✅ **Dashboard Running**: `http://localhost:3000`
✅ **Mock Data Mode**: Full functionality without backend
✅ **System Control**: Ready for backend management
✅ **API Tester**: Ready for endpoint testing
✅ **Real-time Monitoring**: Live status updates

## Next Steps

1. **Start System Controller**: `python start_system_controller.py`
2. **Use System Control** to start backend services
3. **Switch to live data** by setting `REACT_APP_USE_MOCK=false`
4. **Test complete workflow** from data collection to portfolio analysis

The dashboard is now a complete operational control center for the Investment MCP system!