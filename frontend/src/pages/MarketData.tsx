import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  IconButton,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Remove as NeutralIcon,
  AccountBalance as BankIcon,
  Home as HomeIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  BarElement,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { useQuery } from '@tanstack/react-query';
import { apiService, queryKeys } from '../services/api';
import LoadingCard from '../components/LoadingCard';
import ErrorCard from '../components/ErrorCard';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const MarketData: React.FC = () => {
  // Future enhancement: period selection
  // const [selectedPeriod, setSelectedPeriod] = useState('1y');

  // Market Sentiment
  const { 
    data: marketSentiment, 
    isLoading: sentimentLoading, 
    error: sentimentError,
    refetch: refetchSentiment 
  } = useQuery({
    queryKey: queryKeys.market.sentiment,
    queryFn: apiService.getMarketSentiment,
  });

  // Economic Indicators
  const { 
    data: economicIndicators, 
    isLoading: economicLoading, 
    error: economicError,
    refetch: refetchEconomic 
  } = useQuery({
    queryKey: queryKeys.economic.indicators,
    queryFn: apiService.getSwedishEconomicIndicators,
  });

  // Interest Rates
  const { 
    data: interestRates, 
    isLoading: ratesLoading, 
    error: ratesError,
    refetch: refetchRates 
  } = useQuery({
    queryKey: queryKeys.economic.interestRates,
    queryFn: apiService.getInterestRates,
  });

  // Housing Data (future enhancement)
  const { 
    isLoading: housingLoading, 
    error: housingError,
    refetch: refetchHousing 
  } = useQuery({
    queryKey: queryKeys.economic.housing,
    queryFn: apiService.getHousingData,
  });

  // Future utility function
  // const formatPercentage = (value: number) => {
  //   return `${(value * 100).toFixed(2)}%`;
  // };

  const getTrendIcon = (change: number) => {
    if (change > 0) return <TrendingUpIcon color="success" />;
    if (change < 0) return <TrendingDownIcon color="error" />;
    return <NeutralIcon color="disabled" />;
  };

  const getTrendColor = (change: number) => {
    if (change > 0) return 'success';
    if (change < 0) return 'error';
    return 'default';
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive':
      case 'optimistic':
        return 'success';
      case 'negative':
      case 'pessimistic':
        return 'error';
      case 'neutral':
      case 'cautious':
        return 'warning';
      default:
        return 'default';
    }
  };

  // Mock chart data for demonstration
  const economicTrendData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [
      {
        label: 'GDP Growth',
        data: [2.1, 2.3, 2.2, 2.4, 2.6, 2.5],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.4,
      },
      {
        label: 'Inflation Rate',
        data: [1.8, 2.1, 2.0, 1.9, 2.2, 2.1],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.4,
      },
    ],
  };

  const sectorPerformanceData = {
    labels: ['Technology', 'Finance', 'Healthcare', 'Energy', 'Real Estate'],
    datasets: [
      {
        label: 'YTD Performance (%)',
        data: [12.5, 8.3, 6.7, -2.1, 4.2],
        backgroundColor: [
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 206, 86, 0.8)',
          'rgba(75, 192, 192, 0.8)',
          'rgba(255, 99, 132, 0.8)',
          'rgba(153, 102, 255, 0.8)',
        ],
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Market Data
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Real-time Swedish market conditions and economic indicators
      </Typography>

      <Grid container spacing={3}>
        {/* Market Sentiment */}
        <Grid item xs={12} md={6}>
          {sentimentLoading ? (
            <LoadingCard title={true} />
          ) : sentimentError ? (
            <ErrorCard 
              title="Market Sentiment" 
              error={sentimentError} 
              onRetry={refetchSentiment}
            />
          ) : (
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    Market Sentiment
                  </Typography>
                  <IconButton size="small" onClick={() => refetchSentiment()}>
                    <RefreshIcon />
                  </IconButton>
                </Box>
                
                {marketSentiment && (
                  <>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <AssessmentIcon sx={{ mr: 1 }} />
                      <Chip
                        label={marketSentiment.overall_sentiment.replace('_', ' ').toUpperCase()}
                        color={getSentimentColor(marketSentiment.overall_sentiment)}
                        sx={{ mr: 1 }}
                      />
                      <Typography variant="body2" color="text.secondary">
                        Score: {(marketSentiment.sentiment_score * 100).toFixed(0)}%
                      </Typography>
                    </Box>

                    <Typography variant="subtitle2" gutterBottom>
                      Key Market Factors:
                    </Typography>
                    <Box component="ul" sx={{ pl: 2, mb: 2 }}>
                      {marketSentiment.key_factors.map((factor, index) => (
                        <li key={index}>
                          <Typography variant="body2">{factor}</Typography>
                        </li>
                      ))}
                    </Box>

                    <Typography variant="subtitle2" gutterBottom>
                      Sector Sentiment:
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                      {Object.entries(marketSentiment.sector_sentiment).map(([sector, sentiment]) => (
                        <Chip
                          key={sector}
                          label={`${sector}: ${sentiment}`}
                          size="small"
                          color={getSentimentColor(sentiment)}
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Economic Indicators */}
        <Grid item xs={12} md={6}>
          {economicLoading ? (
            <LoadingCard title={true} />
          ) : economicError ? (
            <ErrorCard 
              title="Economic Indicators" 
              error={economicError} 
              onRetry={refetchEconomic}
            />
          ) : (
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    Economic Indicators
                  </Typography>
                  <IconButton size="small" onClick={() => refetchEconomic()}>
                    <RefreshIcon />
                  </IconButton>
                </Box>
                
                {economicIndicators && economicIndicators.length > 0 ? (
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Indicator</TableCell>
                          <TableCell align="right">Value</TableCell>
                          <TableCell align="right">Change</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {economicIndicators.slice(0, 5).map((indicator, index) => (
                          <TableRow key={index}>
                            <TableCell>{indicator.indicator}</TableCell>
                            <TableCell align="right">{indicator.value}</TableCell>
                            <TableCell align="right">
                              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                {getTrendIcon(indicator.change)}
                                <Chip
                                  label={`${indicator.change > 0 ? '+' : ''}${indicator.change}%`}
                                  size="small"
                                  color={getTrendColor(indicator.change)}
                                  sx={{ ml: 1 }}
                                />
                              </Box>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No economic data available
                  </Typography>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Interest Rates */}
        <Grid item xs={12} md={6}>
          {ratesLoading ? (
            <LoadingCard title={true} />
          ) : ratesError ? (
            <ErrorCard 
              title="Interest Rates" 
              error={ratesError} 
              onRetry={refetchRates}
            />
          ) : (
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <BankIcon sx={{ mr: 1 }} />
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    Swedish Interest Rates
                  </Typography>
                  <IconButton size="small" onClick={() => refetchRates()}>
                    <RefreshIcon />
                  </IconButton>
                </Box>
                
                {interestRates ? (
                  <Box>
                    {/* Mock interest rate data */}
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Riksbank Policy Rate
                      </Typography>
                      <Typography variant="h4" color="primary">
                        3.75%
                      </Typography>
                      <Chip
                        label="+0.25%"
                        size="small"
                        color="warning"
                        sx={{ mt: 1 }}
                      />
                    </Box>
                    
                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableBody>
                          <TableRow>
                            <TableCell>10-Year Government Bond</TableCell>
                            <TableCell align="right">2.45%</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Mortgage Rate (3 months)</TableCell>
                            <TableCell align="right">4.12%</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Corporate Bond Rate</TableCell>
                            <TableCell align="right">3.89%</TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No interest rate data available
                  </Typography>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Housing Market */}
        <Grid item xs={12} md={6}>
          {housingLoading ? (
            <LoadingCard title={true} />
          ) : housingError ? (
            <ErrorCard 
              title="Housing Market" 
              error={housingError} 
              onRetry={refetchHousing}
            />
          ) : (
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <HomeIcon sx={{ mr: 1 }} />
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    Swedish Housing Market
                  </Typography>
                  <IconButton size="small" onClick={() => refetchHousing()}>
                    <RefreshIcon />
                  </IconButton>
                </Box>
                
                {/* Mock housing data */}
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        Average Price
                      </Typography>
                      <Typography variant="h6">
                        4.2M SEK
                      </Typography>
                      <Chip
                        label="+2.5%"
                        size="small"
                        color="success"
                        sx={{ mt: 1 }}
                      />
                    </Paper>
                  </Grid>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        Price/sqm
                      </Typography>
                      <Typography variant="h6">
                        62,000 SEK
                      </Typography>
                      <Chip
                        label="+1.8%"
                        size="small"
                        color="success"
                        sx={{ mt: 1 }}
                      />
                    </Paper>
                  </Grid>
                </Grid>

                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Regional Performance (YTD)
                  </Typography>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableBody>
                        <TableRow>
                          <TableCell>Stockholm</TableCell>
                          <TableCell align="right">+3.2%</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Gothenburg</TableCell>
                          <TableCell align="right">+2.1%</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Malmö</TableCell>
                          <TableCell align="right">+1.8%</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Economic Trends Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Economic Trends
              </Typography>
              <Box sx={{ height: 300 }}>
                <Line data={economicTrendData} options={chartOptions} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Sector Performance Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Sector Performance
              </Typography>
              <Box sx={{ height: 300 }}>
                <Bar data={sectorPerformanceData} options={chartOptions} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MarketData;