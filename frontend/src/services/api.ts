import axios, { AxiosInstance, AxiosResponse } from 'axios';
import * as mockData from './mockApi';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_KEY = process.env.REACT_APP_API_KEY || 'dev-key-123';
const USE_MOCK_DATA = process.env.REACT_APP_USE_MOCK === 'true' || false;

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API Service Interface Types
export interface HealthStatus {
  status: string;
  version: string;
  timestamp: string;
}

export interface FundInfo {
  id: string;
  name: string;
  category: string;
  risk_level: string;
  description: string;
  inception_date: string;
  currency: string;
  expense_ratio: number;
}

export interface PortfolioAnalysisRequest {
  risk_tolerance: 'low' | 'medium' | 'high' | 'very_high';
  investment_horizon: 'short' | 'medium' | 'long';
  investment_amount: number;
  current_portfolio?: Record<string, number>;
}

export interface PortfolioAnalysisResponse {
  portfolio_id: string;
  recommended_allocation: Record<string, number>;
  expected_return: number;
  expected_risk: number;
  sharpe_ratio: number;
  risk_metrics: {
    value_at_risk: number;
    max_drawdown: number;
    volatility: number;
  };
  ai_insights: {
    summary: string;
    key_recommendations: string[];
    risk_assessment: string;
  };
}

export interface EconomicIndicator {
  indicator: string;
  value: number;
  change: number;
  timestamp: string;
}

export interface MarketSentiment {
  overall_sentiment: string;
  sentiment_score: number;
  key_factors: string[];
  sector_sentiment: Record<string, string>;
}

// API Service Class
class ApiService {
  // Health Check
  async getHealth(): Promise<HealthStatus> {
    if (USE_MOCK_DATA) {
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 500));
      return mockData.mockHealthData;
    }
    
    try {
      const response = await apiClient.get('/api/v1/health');
      return response.data;
    } catch (error) {
      // Fallback to mock data if API is unavailable
      console.warn('API unavailable, using mock data');
      return mockData.mockHealthData;
    }
  }

  // System Status
  async getSystemStatus(): Promise<any> {
    if (USE_MOCK_DATA) {
      await new Promise(resolve => setTimeout(resolve, 300));
      return mockData.mockSystemStatus;
    }
    
    try {
      const response = await apiClient.get('/');
      return response.data;
    } catch (error) {
      console.warn('API unavailable, using mock data');
      return mockData.mockSystemStatus;
    }
  }

  // Portfolio Analysis
  async analyzePortfolio(request: PortfolioAnalysisRequest): Promise<PortfolioAnalysisResponse> {
    if (USE_MOCK_DATA) {
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate analysis time
      return mockData.mockPortfolioAnalysis;
    }
    
    try {
      const response = await apiClient.post('/api/v1/portfolio/analysis', request);
      return response.data;
    } catch (error) {
      console.warn('API unavailable, using mock data');
      await new Promise(resolve => setTimeout(resolve, 1000));
      return mockData.mockPortfolioAnalysis;
    }
  }

  async stressTestPortfolio(portfolio: Record<string, number>): Promise<any> {
    const response = await apiClient.post('/api/v1/portfolio/stress-test', {
      portfolio_allocation: portfolio
    });
    return response.data;
  }

  // Fund Data
  async getAllFunds(): Promise<FundInfo[]> {
    if (USE_MOCK_DATA) {
      await new Promise(resolve => setTimeout(resolve, 800));
      return mockData.mockFunds;
    }
    
    try {
      const response = await apiClient.get('/api/v1/funds');
      return response.data.funds;
    } catch (error) {
      console.warn('API unavailable, using mock data');
      return mockData.mockFunds;
    }
  }

  async getFundDetails(fundId: string): Promise<FundInfo> {
    const response = await apiClient.get(`/api/v1/funds/${fundId}`);
    return response.data;
  }

  async getFundPerformance(fundId: string, period: string = '1y'): Promise<any> {
    const response = await apiClient.get(`/api/v1/funds/${fundId}/performance`, {
      params: { period }
    });
    return response.data;
  }

  // Economic Data
  async getSwedishEconomicIndicators(): Promise<EconomicIndicator[]> {
    if (USE_MOCK_DATA) {
      await new Promise(resolve => setTimeout(resolve, 700));
      return mockData.mockEconomicIndicators;
    }
    
    try {
      const response = await apiClient.get('/api/v1/economic/sweden/indicators');
      return response.data.indicators;
    } catch (error) {
      console.warn('API unavailable, using mock data');
      return mockData.mockEconomicIndicators;
    }
  }

  async getInterestRates(): Promise<any> {
    const response = await apiClient.get('/api/v1/economic/sweden/interest-rates');
    return response.data;
  }

  async getHousingData(): Promise<any> {
    const response = await apiClient.get('/api/v1/economic/sweden/housing');
    return response.data;
  }

  // Market Analysis
  async getMarketSentiment(): Promise<MarketSentiment> {
    if (USE_MOCK_DATA) {
      await new Promise(resolve => setTimeout(resolve, 600));
      return mockData.mockMarketSentiment;
    }
    
    try {
      const response = await apiClient.get('/api/v1/market/sentiment');
      return response.data;
    } catch (error) {
      console.warn('API unavailable, using mock data');
      return mockData.mockMarketSentiment;
    }
  }

  async getMarketOverview(): Promise<any> {
    const response = await apiClient.get('/api/v1/market/overview');
    return response.data;
  }

  // AI Services
  async getInvestmentRecommendation(params: any): Promise<any> {
    const response = await apiClient.post('/api/v1/ai/investment-recommendation', params);
    return response.data;
  }

  async explainRecommendation(portfolioId: string): Promise<any> {
    const response = await apiClient.get(`/api/v1/ai/explain-recommendation/${portfolioId}`);
    return response.data;
  }
}

// Export singleton instance
export const apiService = new ApiService();

// Export query keys
export { queryKeys } from './queryClient';

export default apiService;