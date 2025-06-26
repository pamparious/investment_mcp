import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Refresh as RefreshIcon,
  OpenInNew as OpenIcon,
  Timeline as TimelineIcon,
  AccountBalance as BankIcon,
  TrendingUp as TrendingIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { apiService, queryKeys } from '../services/api';
import LoadingCard from '../components/LoadingCard';
import ErrorCard from '../components/ErrorCard';

const Dashboard: React.FC = () => {
  // API Health Check
  const { 
    data: healthStatus, 
    isLoading: healthLoading, 
    error: healthError,
    refetch: refetchHealth 
  } = useQuery({
    queryKey: queryKeys.health,
    queryFn: apiService.getHealth,
    refetchInterval: 30000,
  });

  // System Status
  const { 
    data: systemStatus, 
    isLoading: systemLoading, 
    error: systemError,
    refetch: refetchSystem 
  } = useQuery({
    queryKey: queryKeys.systemStatus,
    queryFn: apiService.getSystemStatus,
  });

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

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'online':
        return <CheckIcon color="success" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      default:
        return <ErrorIcon color="error" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'online':
        return 'success';
      case 'warning':
        return 'warning';
      default:
        return 'error';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Investment Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Monitor your Swedish investment portfolio and market conditions
      </Typography>

      <Grid container spacing={3}>
        {/* API Health Status */}
        <Grid item xs={12} md={6}>
          {healthLoading ? (
            <LoadingCard title={true} />
          ) : healthError ? (
            <ErrorCard 
              title="API Health Status" 
              error={healthError} 
              onRetry={refetchHealth}
            />
          ) : (
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    API Health Status
                  </Typography>
                  <IconButton size="small" onClick={() => refetchHealth()}>
                    <RefreshIcon />
                  </IconButton>
                </Box>
                
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  {getStatusIcon(healthStatus?.status || 'error')}
                  <Chip
                    label={healthStatus?.status || 'Unknown'}
                    color={getStatusColor(healthStatus?.status || 'error')}
                    sx={{ ml: 1 }}
                  />
                </Box>
                
                <Typography variant="body2" color="text.secondary">
                  Version: {healthStatus?.version}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Last Check: {new Date(healthStatus?.timestamp || Date.now()).toLocaleString()}
                </Typography>
                
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<OpenIcon />}
                  onClick={() => window.open('http://localhost:8000/docs', '_blank')}
                  sx={{ mt: 2 }}
                >
                  Open API Docs
                </Button>
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* System Overview */}
        <Grid item xs={12} md={6}>
          {systemLoading ? (
            <LoadingCard title={true} />
          ) : systemError ? (
            <ErrorCard 
              title="System Overview" 
              error={systemError} 
              onRetry={refetchSystem}
            />
          ) : (
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    System Overview
                  </Typography>
                  <IconButton size="small" onClick={() => refetchSystem()}>
                    <RefreshIcon />
                  </IconButton>
                </Box>
                
                <List dense>
                  <ListItem>
                    <ListItemIcon>
                      <CheckIcon color="success" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Investment MCP API"
                      secondary={`Version ${systemStatus?.version || 'Unknown'}`}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <TimelineIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary="Data Collection"
                      secondary="Swedish funds & economic data"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <BankIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary="Portfolio Analysis"
                      secondary="AI-powered optimization"
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          )}
        </Grid>

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
                
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Overall Sentiment
                  </Typography>
                  <Chip
                    label={marketSentiment?.overall_sentiment || 'Unknown'}
                    color={
                      marketSentiment?.overall_sentiment === 'positive' ? 'success' :
                      marketSentiment?.overall_sentiment === 'negative' ? 'error' : 'default'
                    }
                    sx={{ mt: 1 }}
                  />
                </Box>
                
                {marketSentiment?.key_factors && (
                  <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Key Factors
                    </Typography>
                    <List dense>
                      {marketSentiment.key_factors.slice(0, 3).map((factor, index) => (
                        <ListItem key={index} sx={{ pl: 0 }}>
                          <ListItemIcon sx={{ minWidth: 20 }}>
                            <TrendingIcon fontSize="small" />
                          </ListItemIcon>
                          <ListItemText 
                            primary={factor}
                            sx={{ '& .MuiListItemText-primary': { fontSize: '0.875rem' } }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
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
              title="Swedish Economic Indicators" 
              error={economicError} 
              onRetry={refetchEconomic}
            />
          ) : (
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    Swedish Economic Indicators
                  </Typography>
                  <IconButton size="small" onClick={() => refetchEconomic()}>
                    <RefreshIcon />
                  </IconButton>
                </Box>
                
                {economicIndicators && economicIndicators.length > 0 ? (
                  <List dense>
                    {economicIndicators.slice(0, 4).map((indicator, index) => (
                      <ListItem key={index} sx={{ pl: 0 }}>
                        <ListItemText
                          primary={indicator.indicator}
                          secondary={`${indicator.value} (${indicator.change > 0 ? '+' : ''}${indicator.change}%)`}
                        />
                        <Chip
                          label={indicator.change > 0 ? '↑' : indicator.change < 0 ? '↓' : '→'}
                          color={indicator.change > 0 ? 'success' : indicator.change < 0 ? 'error' : 'default'}
                          size="small"
                        />
                      </ListItem>
                    ))}
                  </List>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No economic data available
                  </Typography>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  onClick={() => window.location.href = '/portfolio'}
                >
                  Analyze Portfolio
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => window.location.href = '/funds'}
                >
                  Browse Funds
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => window.location.href = '/market'}
                >
                  Market Data
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => window.location.href = '/system'}
                >
                  System Control
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => window.location.href = '/api-tester'}
                >
                  API Tester
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => window.open('http://localhost:8000/docs', '_blank')}
                >
                  API Documentation
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;