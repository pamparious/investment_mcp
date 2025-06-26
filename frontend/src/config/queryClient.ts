/**
 * Optimized React Query client configuration
 */

import { QueryClient, DefaultOptions } from '@tanstack/react-query';
import { ApiError } from '../types/api';

// Custom error handler for React Query
const errorHandler = (error: unknown): string => {
  if (error && typeof error === 'object' && 'response' in error) {
    const apiError = error as { response?: { data?: ApiError; status?: number } };
    if (apiError.response?.data?.message) {
      return apiError.response.data.message;
    }
    if (apiError.response?.status) {
      return `API Error: ${apiError.response.status}`;
    }
  }
  
  if (error instanceof Error) {
    return error.message;
  }
  
  return 'An unexpected error occurred';
};

// Default query options optimized for performance
const queryConfig: DefaultOptions = {
  queries: {
    // Cache data for 5 minutes by default
    staleTime: 5 * 60 * 1000,
    
    // Keep data in cache for 10 minutes
    gcTime: 10 * 60 * 1000,
    
    // Retry failed requests up to 3 times with exponential backoff
    retry: (failureCount, error) => {
      // Don't retry for 4xx errors (client errors)
      if (error && typeof error === 'object' && 'response' in error) {
        const apiError = error as { response?: { status?: number } };
        if (apiError.response?.status && apiError.response.status >= 400 && apiError.response.status < 500) {
          return false;
        }
      }
      return failureCount < 3;
    },
    
    // Exponential backoff delay
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    
    // Refetch on window focus for critical data only
    refetchOnWindowFocus: false,
    
    // Don't refetch on reconnect by default
    refetchOnReconnect: 'always',
    
    // Error handling moved to global error handler
  },
  mutations: {
    // Retry mutations once
    retry: 1,
    
    // Error handling for mutations moved to global handler
  },
};

// Create optimized query client
export const createQueryClient = (): QueryClient => {
  return new QueryClient({
    defaultOptions: queryConfig,
  });
};

// Specific query configurations for different data types
export const queryConfigs = {
  // Real-time data (frequent updates)
  realtime: {
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 30 * 1000,
    refetchOnWindowFocus: true,
  },
  
  // System status (medium frequency)
  systemStatus: {
    staleTime: 2 * 60 * 1000, // 2 minutes
    refetchInterval: 2 * 60 * 1000,
    refetchOnWindowFocus: true,
  },
  
  // Market data (moderate frequency)
  marketData: {
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
  },
  
  // Economic indicators (low frequency)
  economicData: {
    staleTime: 30 * 60 * 1000, // 30 minutes
    refetchInterval: 30 * 60 * 1000,
    refetchOnWindowFocus: false,
  },
  
  // Fund data (very low frequency)
  fundData: {
    staleTime: 60 * 60 * 1000, // 1 hour
    refetchInterval: false,
    refetchOnWindowFocus: false,
  },
  
  // Static data (rarely changes)
  staticData: {
    staleTime: 24 * 60 * 60 * 1000, // 24 hours
    refetchInterval: false,
    refetchOnWindowFocus: false,
  },
};

// Query key factory for consistent key management
export const queryKeys = {
  // Health and system status
  health: ['health'] as const,
  systemStatus: ['system', 'status'] as const,
  
  // Fund-related queries
  funds: {
    all: ['funds'] as const,
    byId: (id: string) => ['funds', id] as const,
    byCategory: (category: string) => ['funds', 'category', category] as const,
    performance: (id: string) => ['funds', id, 'performance'] as const,
  },
  
  // Portfolio queries
  portfolio: {
    analysis: ['portfolio', 'analysis'] as const,
    stressTest: ['portfolio', 'stress-test'] as const,
    optimization: ['portfolio', 'optimization'] as const,
  },
  
  // Market data queries
  market: {
    sentiment: ['market', 'sentiment'] as const,
    data: ['market', 'data'] as const,
    trends: ['market', 'trends'] as const,
  },
  
  // Economic data queries
  economic: {
    indicators: ['economic', 'indicators'] as const,
    interestRates: ['economic', 'rates'] as const,
    housing: ['economic', 'housing'] as const,
    sweden: ['economic', 'sweden'] as const,
  },
  
  // System control queries
  system: {
    status: ['system', 'status'] as const,
    logs: (component?: string) => ['system', 'logs', ...(component ? [component] : [])] as const,
    processes: ['system', 'processes'] as const,
    database: ['system', 'database'] as const,
  },
  
  // MCP queries
  mcp: {
    tools: ['mcp', 'tools'] as const,
    jobs: ['mcp', 'jobs'] as const,
    status: ['mcp', 'status'] as const,
  },
};