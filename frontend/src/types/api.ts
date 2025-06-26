/**
 * Comprehensive TypeScript interfaces for Investment MCP API
 */

// Base API Response structure
export interface ApiResponse<T = any> {
  data: T;
  status: 'success' | 'error';
  message?: string;
  timestamp: string;
}

// Error handling types
export interface ApiError {
  status: number;
  message: string;
  code?: string;
  details?: Record<string, unknown>;
}

// Health and Status types
export interface HealthStatus {
  status: 'healthy' | 'warning' | 'error';
  version: string;
  timestamp: string;
  database_status?: 'connected' | 'disconnected' | 'error';
  services?: ServiceStatus[];
}

export interface ServiceStatus {
  name: string;
  status: 'running' | 'stopped' | 'error';
  uptime?: string;
  port?: number;
  pid?: number;
}

// System Controller types
export interface SystemStatus {
  backend_api: ServiceInfo;
  mcp_server: ServiceInfo;
  database: DatabaseInfo;
  data_collectors: DataCollectorInfo;
}

export interface ServiceInfo {
  status: 'running' | 'stopped' | 'error';
  port: number;
  pid?: number;
  uptime?: string;
  tools_available?: number;
}

export interface DatabaseInfo {
  status: 'connected' | 'disconnected' | 'error';
  path: string;
  size: string;
  tables: string[];
  last_updated?: string;
  error?: string;
}

export interface DataCollectorInfo {
  status: 'idle' | 'running' | 'error';
  last_run: string;
  collected_records: number;
}

// Portfolio Analysis types
export interface PortfolioAnalysisRequest {
  risk_tolerance: 'low' | 'medium' | 'high';
  investment_horizon: 'short' | 'medium' | 'long';
  monthly_investment?: number;
  current_age?: number;
  target_retirement_age?: number;
  investment_goals?: string[];
  existing_portfolio?: Record<string, number>;
}

export interface PortfolioAnalysisResponse {
  recommended_allocation: Record<string, number>;
  expected_return: number;
  risk_metrics: RiskMetrics;
  ai_insights: AIInsights;
  stress_test_results?: StressTestResult[];
  rebalancing_recommendations?: RebalancingRecommendation[];
}

export interface RiskMetrics {
  sharpe_ratio: number;
  max_drawdown: number;
  volatility: number;
  var_95: number;
  beta: number;
}

export interface AIInsights {
  summary: string;
  key_recommendations: string[];
  risk_assessment: string;
  market_outlook: string;
  confidence_score: number;
}

export interface StressTestResult {
  scenario: string;
  portfolio_impact: number;
  probability: number;
  description: string;
}

export interface RebalancingRecommendation {
  fund: string;
  current_allocation: number;
  target_allocation: number;
  action: 'buy' | 'sell' | 'hold';
  amount: number;
}

// Fund types
export interface Fund {
  id: string;
  name: string;
  category: 'global_equity' | 'emerging_markets' | 'europe' | 'nordic' | 'sweden' | 'usa' | 'japan' | 'small_cap' | 'commodities' | 'crypto' | 'real_estate';
  risk_level: 'low' | 'medium' | 'high';
  annual_fee: number;
  description: string;
  performance: PerformanceData;
  holdings?: Holding[];
}

export interface PerformanceData {
  ytd: number;
  one_year: number;
  three_year: number;
  five_year: number;
  inception: number;
  last_updated: string;
}

export interface Holding {
  name: string;
  weight: number;
  sector?: string;
  country?: string;
}

// Market Data types
export interface MarketSentiment {
  overall_sentiment: 'positive' | 'negative' | 'neutral';
  confidence_score: number;
  key_factors: string[];
  sentiment_breakdown: Record<string, number>;
  last_updated: string;
}

export interface EconomicIndicator {
  indicator: string;
  value: number;
  change: number;
  unit: string;
  last_updated: string;
  source: 'riksbank' | 'scb' | 'external';
}

export interface InterestRate {
  rate_type: string;
  rate: number;
  change: number;
  effective_date: string;
  next_meeting?: string;
}

export interface HousingData {
  region: string;
  price_index: number;
  monthly_change: number;
  yearly_change: number;
  average_price: number;
  transactions: number;
  last_updated: string;
}

// MCP Tool types
export interface MCPTool {
  name: string;
  description: string;
  category: 'data_collection' | 'analysis' | 'optimization' | 'monitoring';
  status: 'available' | 'running' | 'error';
  last_used?: string;
  usage_count?: number;
}

export interface DataCollectionJob {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  source: 'yfinance' | 'riksbank' | 'scb';
  progress: number;
  records_collected: number;
  started_at?: string;
  completed_at?: string;
  error?: string;
}

export interface LogEntry {
  timestamp: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  component: string;
  message: string;
  metadata?: Record<string, unknown>;
}

// API Testing types
export interface APIEndpoint {
  name: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  description: string;
  parameters?: APIParameter[];
  body_schema?: Record<string, unknown>;
  response_schema?: Record<string, unknown>;
}

export interface APIParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'array' | 'object';
  required: boolean;
  description: string;
  default?: unknown;
  example?: unknown;
}

export interface APITestResult {
  endpoint: string;
  method: string;
  status_code: number;
  response_time: number;
  response_data: unknown;
  error?: string;
  timestamp: string;
}

// Query Key types for React Query
export interface QueryKeys {
  health: string[];
  systemStatus: string[];
  funds: {
    all: string[];
    byId: (id: string) => string[];
    byCategory: (category: string) => string[];
  };
  portfolio: {
    analysis: string[];
    stressTest: string[];
  };
  market: {
    sentiment: string[];
    data: string[];
  };
  economic: {
    indicators: string[];
    interestRates: string[];
    housing: string[];
  };
}

// Configuration types
export interface ApiConfig {
  baseURL: string;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
  enableMockData: boolean;
}