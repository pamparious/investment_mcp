import { QueryClient } from '@tanstack/react-query';

// Configure React Query
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
      retry: (failureCount, error: any) => {
        // Don't retry on 4xx errors (client errors)
        if (error?.response?.status >= 400 && error?.response?.status < 500) {
          return false;
        }
        // Retry up to 3 times for other errors
        return failureCount < 3;
      },
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});

// Query Keys - Centralized query key factory
export const queryKeys = {
  health: ['health'],
  systemStatus: ['system', 'status'],
  
  // Portfolio queries
  portfolio: {
    analysis: (params: any) => ['portfolio', 'analysis', params],
    stressTest: (portfolio: Record<string, number>) => ['portfolio', 'stress-test', portfolio],
  },
  
  // Fund queries
  funds: {
    all: ['funds', 'all'],
    details: (fundId: string) => ['funds', 'details', fundId],
    performance: (fundId: string, period: string) => ['funds', 'performance', fundId, period],
  },
  
  // Economic data queries
  economic: {
    indicators: ['economic', 'sweden', 'indicators'],
    interestRates: ['economic', 'sweden', 'interest-rates'],
    housing: ['economic', 'sweden', 'housing'],
  },
  
  // Market queries
  market: {
    sentiment: ['market', 'sentiment'],
    overview: ['market', 'overview'],
  },
  
  // AI queries
  ai: {
    recommendation: (params: any) => ['ai', 'recommendation', params],
    explanation: (portfolioId: string) => ['ai', 'explanation', portfolioId],
  },
};