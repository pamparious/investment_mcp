import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Analytics as AnalyticsIcon,
  TrendingUp as TrendingUpIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';
import { useMutation } from '@tanstack/react-query';
import { apiService, PortfolioAnalysisRequest } from '../services/api';

const PortfolioAnalysis: React.FC = () => {
  const [formData, setFormData] = useState<PortfolioAnalysisRequest>({
    risk_tolerance: 'medium',
    investment_horizon: 'medium',
    investment_amount: 100000,
  });

  // Portfolio Analysis Mutation
  const analysisMutation = useMutation({
    mutationFn: (data: PortfolioAnalysisRequest) => apiService.analyzePortfolio(data),
    onSuccess: (data) => {
      console.log('Analysis completed:', data);
    },
    onError: (error) => {
      console.error('Analysis failed:', error);
    },
  });

  const handleInputChange = (field: keyof PortfolioAnalysisRequest, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    analysisMutation.mutate(formData);
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('sv-SE', {
      style: 'currency',
      currency: 'SEK',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const getRiskToleranceColor = (level: string) => {
    switch (level) {
      case 'low': return 'success';
      case 'medium': return 'info';
      case 'high': return 'warning';
      case 'very_high': return 'error';
      default: return 'default';
    }
  };

  const getHorizonColor = (horizon: string) => {
    switch (horizon) {
      case 'short': return 'error';
      case 'medium': return 'warning';
      case 'long': return 'success';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Portfolio Analysis
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Get AI-powered portfolio recommendations based on your investment profile
      </Typography>

      <Grid container spacing={3}>
        {/* Analysis Form */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Investment Profile
              </Typography>
              
              <form onSubmit={handleSubmit}>
                <Grid container spacing={3}>
                  {/* Risk Tolerance */}
                  <Grid item xs={12}>
                    <FormControl fullWidth>
                      <InputLabel>Risk Tolerance</InputLabel>
                      <Select
                        value={formData.risk_tolerance}
                        label="Risk Tolerance"
                        onChange={(e) => handleInputChange('risk_tolerance', e.target.value)}
                      >
                        <MenuItem value="low">Low - Conservative</MenuItem>
                        <MenuItem value="medium">Medium - Balanced</MenuItem>
                        <MenuItem value="high">High - Growth-oriented</MenuItem>
                        <MenuItem value="very_high">Very High - Aggressive</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>

                  {/* Investment Horizon */}
                  <Grid item xs={12}>
                    <FormControl fullWidth>
                      <InputLabel>Investment Horizon</InputLabel>
                      <Select
                        value={formData.investment_horizon}
                        label="Investment Horizon"
                        onChange={(e) => handleInputChange('investment_horizon', e.target.value)}
                      >
                        <MenuItem value="short">Short-term (1-3 years)</MenuItem>
                        <MenuItem value="medium">Medium-term (3-10 years)</MenuItem>
                        <MenuItem value="long">Long-term (10+ years)</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>

                  {/* Investment Amount */}
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Investment Amount (SEK)"
                      type="number"
                      value={formData.investment_amount}
                      onChange={(e) => handleInputChange('investment_amount', Number(e.target.value))}
                      InputProps={{
                        inputProps: { min: 1000, step: 1000 }
                      }}
                    />
                  </Grid>

                  {/* Submit Button */}
                  <Grid item xs={12}>
                    <Button
                      type="submit"
                      variant="contained"
                      fullWidth
                      size="large"
                      disabled={analysisMutation.isPending}
                      startIcon={
                        analysisMutation.isPending ? 
                        <CircularProgress size={20} /> : 
                        <AnalyticsIcon />
                      }
                    >
                      {analysisMutation.isPending ? 'Analyzing...' : 'Analyze Portfolio'}
                    </Button>
                  </Grid>
                </Grid>
              </form>
            </CardContent>
          </Card>
        </Grid>

        {/* Profile Summary */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Profile Summary
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Risk Tolerance
                </Typography>
                <Chip
                  label={formData.risk_tolerance.replace('_', ' ').toUpperCase()}
                  color={getRiskToleranceColor(formData.risk_tolerance)}
                  sx={{ mt: 0.5 }}
                />
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Investment Horizon
                </Typography>
                <Chip
                  label={formData.investment_horizon.replace('_', ' ').toUpperCase()}
                  color={getHorizonColor(formData.investment_horizon)}
                  sx={{ mt: 0.5 }}
                />
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Investment Amount
                </Typography>
                <Typography variant="h6" color="primary">
                  {formatCurrency(formData.investment_amount)}
                </Typography>
              </Box>

              <Alert severity="info" sx={{ mt: 2 }}>
                Analysis will provide personalized recommendations for Swedish investment funds
                based on your profile and current market conditions.
              </Alert>
            </CardContent>
          </Card>
        </Grid>

        {/* Error Display */}
        {analysisMutation.isError && (
          <Grid item xs={12}>
            <Alert severity="error">
              Analysis failed: {(analysisMutation.error as any)?.response?.data?.error || 'Unknown error'}
            </Alert>
          </Grid>
        )}

        {/* Results */}
        {analysisMutation.data && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                  <TrendingUpIcon sx={{ mr: 1 }} />
                  Portfolio Recommendation
                </Typography>

                {/* Key Metrics */}
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        Expected Return
                      </Typography>
                      <Typography variant="h6" color="success.main">
                        {formatPercentage(analysisMutation.data.expected_return)}
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        Expected Risk
                      </Typography>
                      <Typography variant="h6" color="warning.main">
                        {formatPercentage(analysisMutation.data.expected_risk)}
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        Sharpe Ratio
                      </Typography>
                      <Typography variant="h6">
                        {analysisMutation.data.sharpe_ratio.toFixed(2)}
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        Portfolio ID
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {analysisMutation.data.portfolio_id.slice(0, 8)}...
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>

                {/* Recommended Allocation */}
                <Accordion defaultExpanded>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="h6">Recommended Allocation</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <TableContainer component={Paper} variant="outlined">
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Fund</TableCell>
                            <TableCell align="right">Allocation</TableCell>
                            <TableCell align="right">Amount (SEK)</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(analysisMutation.data.recommended_allocation).map(([fundId, allocation]) => (
                            <TableRow key={fundId}>
                              <TableCell>{fundId}</TableCell>
                              <TableCell align="right">
                                <Chip 
                                  label={formatPercentage(allocation as number)} 
                                  size="small"
                                  color="primary"
                                  variant="outlined"
                                />
                              </TableCell>
                              <TableCell align="right">
                                {formatCurrency(formData.investment_amount * (allocation as number))}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </AccordionDetails>
                </Accordion>

                {/* Risk Metrics */}
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="h6">Risk Analysis</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2 }}>
                          <Typography variant="body2" color="text.secondary">
                            Value at Risk (95%)
                          </Typography>
                          <Typography variant="h6" color="error.main">
                            {formatPercentage(analysisMutation.data.risk_metrics.value_at_risk)}
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2 }}>
                          <Typography variant="body2" color="text.secondary">
                            Maximum Drawdown
                          </Typography>
                          <Typography variant="h6" color="error.main">
                            {formatPercentage(analysisMutation.data.risk_metrics.max_drawdown)}
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2 }}>
                          <Typography variant="body2" color="text.secondary">
                            Volatility
                          </Typography>
                          <Typography variant="h6" color="warning.main">
                            {formatPercentage(analysisMutation.data.risk_metrics.volatility)}
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>

                {/* AI Insights */}
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
                      <AssessmentIcon sx={{ mr: 1 }} />
                      AI Insights
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Alert severity="info" sx={{ mb: 2 }}>
                      <Typography variant="body2">
                        {analysisMutation.data.ai_insights.summary}
                      </Typography>
                    </Alert>
                    
                    <Typography variant="subtitle2" gutterBottom>
                      Key Recommendations:
                    </Typography>
                    <Box component="ul" sx={{ pl: 2 }}>
                      {analysisMutation.data.ai_insights.key_recommendations.map((rec, index) => (
                        <li key={index}>
                          <Typography variant="body2">{rec}</Typography>
                        </li>
                      ))}
                    </Box>
                    
                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                      Risk Assessment:
                    </Typography>
                    <Typography variant="body2">
                      {analysisMutation.data.ai_insights.risk_assessment}
                    </Typography>
                  </AccordionDetails>
                </Accordion>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default PortfolioAnalysis;