// System control API for managing the Investment MCP system
import axios from 'axios';

const SYSTEM_API_BASE = 'http://localhost:8001'; // System control API port

export interface SystemStatus {
  backend_api: {
    status: 'running' | 'stopped' | 'error';
    port: number;
    pid?: number;
    uptime?: string;
  };
  mcp_server: {
    status: 'running' | 'stopped' | 'error';
    port: number;
    pid?: number;
    tools_available?: number;
  };
  database: {
    status: 'connected' | 'disconnected' | 'error';
    path: string;
    size?: string;
    last_updated?: string;
  };
  data_collectors: {
    last_run?: string;
    next_scheduled?: string;
    status: 'idle' | 'running' | 'error';
    collected_records?: number;
  };
}

export interface LogEntry {
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  component: string;
  message: string;
}

export interface MCPTool {
  name: string;
  description: string;
  last_used?: string;
  usage_count?: number;
  status: 'available' | 'error';
}

export interface DataCollectionJob {
  id: string;
  source: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  started_at?: string;
  completed_at?: string;
  records_collected?: number;
  error?: string;
}

class SystemApiService {
  private baseURL = SYSTEM_API_BASE;

  // System Status
  async getSystemStatus(): Promise<SystemStatus> {
    try {
      const response = await axios.get(`${this.baseURL}/system/status`);
      return response.data;
    } catch (error) {
      // Return mock data if system API is not available
      return this.getMockSystemStatus();
    }
  }

  // Backend API Control
  async startBackendAPI(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await axios.post(`${this.baseURL}/system/backend/start`);
      return response.data;
    } catch (error) {
      return { success: false, message: 'System API not available - cannot control backend' };
    }
  }

  async stopBackendAPI(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await axios.post(`${this.baseURL}/system/backend/stop`);
      return response.data;
    } catch (error) {
      return { success: false, message: 'System API not available - cannot control backend' };
    }
  }

  async restartBackendAPI(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await axios.post(`${this.baseURL}/system/backend/restart`);
      return response.data;
    } catch (error) {
      return { success: false, message: 'System API not available - cannot control backend' };
    }
  }

  // MCP Server Control
  async startMCPServer(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await axios.post(`${this.baseURL}/system/mcp/start`);
      return response.data;
    } catch (error) {
      return { success: false, message: 'System API not available - cannot control MCP server' };
    }
  }

  async stopMCPServer(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await axios.post(`${this.baseURL}/system/mcp/stop`);
      return response.data;
    } catch (error) {
      return { success: false, message: 'System API not available - cannot control MCP server' };
    }
  }

  async restartMCPServer(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await axios.post(`${this.baseURL}/system/mcp/restart`);
      return response.data;
    } catch (error) {
      return { success: false, message: 'System API not available - cannot control MCP server' };
    }
  }

  // MCP Tools
  async getMCPTools(): Promise<MCPTool[]> {
    try {
      const response = await axios.get(`${this.baseURL}/system/mcp/tools`);
      return response.data.tools;
    } catch (error) {
      return this.getMockMCPTools();
    }
  }

  async testMCPTool(toolName: string, params: any = {}): Promise<{ success: boolean; result?: any; error?: string }> {
    try {
      const response = await axios.post(`${this.baseURL}/system/mcp/tools/${toolName}/test`, params);
      return response.data;
    } catch (error) {
      return { success: false, error: 'System API not available - cannot test MCP tool' };
    }
  }

  // Data Collection
  async startDataCollection(sources: string[] = ['all']): Promise<{ success: boolean; jobId?: string; message: string }> {
    try {
      const response = await axios.post(`${this.baseURL}/system/data/collect`, { sources });
      return response.data;
    } catch (error) {
      return { success: false, message: 'System API not available - cannot start data collection' };
    }
  }

  async getDataCollectionJobs(): Promise<DataCollectionJob[]> {
    try {
      const response = await axios.get(`${this.baseURL}/system/data/jobs`);
      return response.data.jobs;
    } catch (error) {
      return this.getMockDataJobs();
    }
  }

  async cancelDataCollectionJob(jobId: string): Promise<{ success: boolean; message: string }> {
    try {
      const response = await axios.delete(`${this.baseURL}/system/data/jobs/${jobId}`);
      return response.data;
    } catch (error) {
      return { success: false, message: 'System API not available - cannot cancel job' };
    }
  }

  // System Logs
  async getSystemLogs(component?: string, level?: string, limit: number = 100): Promise<LogEntry[]> {
    try {
      const params = new URLSearchParams();
      if (component) params.append('component', component);
      if (level) params.append('level', level);
      params.append('limit', limit.toString());

      const response = await axios.get(`${this.baseURL}/system/logs?${params}`);
      return response.data.logs;
    } catch (error) {
      return this.getMockLogs();
    }
  }

  // Database Operations
  async getDatabaseInfo(): Promise<{ size: string; tables: string[]; last_backup?: string }> {
    try {
      const response = await axios.get(`${this.baseURL}/system/database/info`);
      return response.data;
    } catch (error) {
      return {
        size: 'Unknown',
        tables: ['market_data', 'economic_data', 'fund_data', 'analysis_results'],
        last_backup: 'Never'
      };
    }
  }

  async backupDatabase(): Promise<{ success: boolean; backup_path?: string; message: string }> {
    try {
      const response = await axios.post(`${this.baseURL}/system/database/backup`);
      return response.data;
    } catch (error) {
      return { success: false, message: 'System API not available - cannot backup database' };
    }
  }

  async resetDatabase(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await axios.post(`${this.baseURL}/system/database/reset`);
      return response.data;
    } catch (error) {
      return { success: false, message: 'System API not available - cannot reset database' };
    }
  }

  // Mock data methods (for when system API is not available)
  private getMockSystemStatus(): SystemStatus {
    return {
      backend_api: {
        status: 'stopped',
        port: 8000,
        uptime: 'Not running'
      },
      mcp_server: {
        status: 'stopped',
        port: 3001,
        tools_available: 0
      },
      database: {
        status: 'disconnected',
        path: './investment_data.db',
        size: 'Unknown'
      },
      data_collectors: {
        status: 'idle',
        last_run: 'Never',
        collected_records: 0
      }
    };
  }

  private getMockMCPTools(): MCPTool[] {
    return [
      {
        name: 'collect_fund_data',
        description: 'Collect historical market data for approved Swedish funds',
        status: 'available',
        usage_count: 0
      },
      {
        name: 'analyze_portfolio',
        description: 'Perform portfolio analysis and optimization',
        status: 'available',
        usage_count: 0
      },
      {
        name: 'get_market_sentiment',
        description: 'Analyze current market sentiment',
        status: 'available',
        usage_count: 0
      },
      {
        name: 'collect_economic_data',
        description: 'Collect Swedish economic indicators',
        status: 'available',
        usage_count: 0
      }
    ];
  }

  private getMockDataJobs(): DataCollectionJob[] {
    return [
      {
        id: 'job-001',
        source: 'yfinance',
        status: 'completed',
        progress: 100,
        started_at: new Date(Date.now() - 3600000).toISOString(),
        completed_at: new Date(Date.now() - 3000000).toISOString(),
        records_collected: 1250
      },
      {
        id: 'job-002',
        source: 'riksbank',
        status: 'running',
        progress: 65,
        started_at: new Date(Date.now() - 1800000).toISOString(),
        records_collected: 89
      }
    ];
  }

  private getMockLogs(): LogEntry[] {
    const now = new Date();
    return [
      {
        timestamp: new Date(now.getTime() - 60000).toISOString(),
        level: 'info',
        component: 'dashboard',
        message: 'Dashboard started in mock mode'
      },
      {
        timestamp: new Date(now.getTime() - 120000).toISOString(),
        level: 'warning',
        component: 'backend',
        message: 'Backend API not responding'
      },
      {
        timestamp: new Date(now.getTime() - 180000).toISOString(),
        level: 'info',
        component: 'mcp',
        message: 'MCP server stopped'
      }
    ];
  }
}

export const systemApiService = new SystemApiService();
export default systemApiService;