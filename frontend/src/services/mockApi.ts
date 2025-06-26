// Mock API for testing the dashboard when backend is not available

export const mockHealthData = {
  status: 'healthy',
  version: '3.0.0',
  timestamp: new Date().toISOString(),
};

export const mockSystemStatus = {
  name: 'Investment MCP API',
  version: '3.0.0',
  description: 'Swedish Investment Analysis API with AI-Powered Recommendations',
  status: 'online',
};

export const mockFunds = [
  {
    id: 'dnb-global-indeks',
    name: 'DNB Global Indeks S',
    category: 'Global Equity',
    risk_level: 'Medium',
    description: 'A global equity index fund tracking world stock markets',
    inception_date: '2010-01-01',
    currency: 'SEK',
    expense_ratio: 0.0045,
  },
  {
    id: 'avanza-emerging-markets',
    name: 'Avanza Emerging Markets',
    category: 'Emerging Markets',
    risk_level: 'High',
    description: 'Emerging markets equity fund focusing on developing economies',
    inception_date: '2012-06-01',
    currency: 'SEK',
    expense_ratio: 0.0075,
  },
  {
    id: 'storebrand-europa',
    name: 'Storebrand Europa A SEK',
    category: 'European Equity',
    risk_level: 'Medium',
    description: 'European equity fund focusing on developed European markets',
    inception_date: '2008-03-01',
    currency: 'SEK',
    expense_ratio: 0.0055,
  },
  {
    id: 'dnb-norden-indeks',
    name: 'DNB Norden Indeks S',
    category: 'Nordic Equity',
    risk_level: 'Medium',
    description: 'Nordic equity index fund covering Scandinavian markets',
    inception_date: '2009-09-01',
    currency: 'SEK',
    expense_ratio: 0.004,
  },
  {
    id: 'plus-allabolag-sverige',
    name: 'PLUS Allabolag Sverige Index',
    category: 'Swedish Equity',
    risk_level: 'Medium',
    description: 'Swedish broad market index fund',
    inception_date: '2011-01-01',
    currency: 'SEK',
    expense_ratio: 0.003,
  },
  {
    id: 'avanza-usa',
    name: 'Avanza USA',
    category: 'US Equity',
    risk_level: 'Medium',
    description: 'US equity fund focusing on American stock markets',
    inception_date: '2010-05-01',
    currency: 'SEK',
    expense_ratio: 0.005,
  },
];

export const mockMarketSentiment = {
  overall_sentiment: 'cautiously_optimistic',
  sentiment_score: 0.65,
  key_factors: [
    'Swedish economic growth outlook positive',
    'Global market volatility elevated',
    'Central bank policy supportive',
  ],
  sector_sentiment: {
    technology: 'positive',
    financials: 'neutral',
    real_estate: 'negative',
    commodities: 'positive',
  },
};

export const mockEconomicIndicators = [
  {
    indicator: 'GDP Growth Rate',
    value: 2.3,
    change: 0.2,
    timestamp: new Date().toISOString(),
  },
  {
    indicator: 'Inflation Rate',
    value: 2.1,
    change: -0.1,
    timestamp: new Date().toISOString(),
  },
  {
    indicator: 'Unemployment Rate',
    value: 7.2,
    change: -0.3,
    timestamp: new Date().toISOString(),
  },
  {
    indicator: 'Interest Rate',
    value: 3.75,
    change: 0.25,
    timestamp: new Date().toISOString(),
  },
];

export const mockPortfolioAnalysis = {
  portfolio_id: 'portfolio-' + Math.random().toString(36).substr(2, 9),
  recommended_allocation: {
    'dnb-global-indeks': 0.25,
    'avanza-usa': 0.20,
    'plus-allabolag-sverige': 0.15,
    'storebrand-europa': 0.15,
    'dnb-norden-indeks': 0.10,
    'avanza-emerging-markets': 0.15,
  },
  expected_return: 0.078,
  expected_risk: 0.142,
  sharpe_ratio: 0.55,
  risk_metrics: {
    value_at_risk: 0.032,
    max_drawdown: 0.18,
    volatility: 0.142,
  },
  ai_insights: {
    summary: 'Based on your medium risk tolerance and medium-term horizon, this diversified portfolio balances growth potential with risk management.',
    key_recommendations: [
      'Maintain core allocation to Swedish and global equities',
      'Diversify across regions to reduce single-market risk',
      'Consider rebalancing quarterly to maintain target allocations',
    ],
    risk_assessment: 'Portfolio risk is appropriate for medium risk tolerance. Expected volatility is within acceptable range for medium-term investing.',
  },
};