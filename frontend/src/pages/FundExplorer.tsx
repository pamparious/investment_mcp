import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  TrendingUp as TrendingUpIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  Assessment as ChartIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { apiService, queryKeys, FundInfo } from '../services/api';
import LoadingCard from '../components/LoadingCard';
import ErrorCard from '../components/ErrorCard';

const FundExplorer: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [riskFilter, setRiskFilter] = useState('all');
  const [selectedFund, setSelectedFund] = useState<FundInfo | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  // Fetch all funds
  const { 
    data: funds, 
    isLoading, 
    error,
    refetch 
  } = useQuery({
    queryKey: queryKeys.funds.all,
    queryFn: apiService.getAllFunds,
  });

  // Fetch fund performance for selected fund
  const { 
    data: performance, 
    isLoading: performanceLoading 
  } = useQuery({
    queryKey: queryKeys.funds.performance(selectedFund?.id || '', '1y'),
    queryFn: () => apiService.getFundPerformance(selectedFund!.id, '1y'),
    enabled: !!selectedFund,
  });

  // Filter funds based on search and filters
  const filteredFunds = funds?.filter(fund => {
    const matchesSearch = fund.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         fund.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = categoryFilter === 'all' || fund.category === categoryFilter;
    const matchesRisk = riskFilter === 'all' || fund.risk_level === riskFilter;
    
    return matchesSearch && matchesCategory && matchesRisk;
  }) || [];

  // Get unique categories and risk levels for filters
  const categories = Array.from(new Set(funds?.map(fund => fund.category) || []));
  const riskLevels = Array.from(new Set(funds?.map(fund => fund.risk_level) || []));

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'low': return 'success';
      case 'medium': return 'warning';
      case 'high': return 'error';
      default: return 'default';
    }
  };

  const getCategoryColor = (category: string) => {
    const colors = ['primary', 'secondary', 'info', 'success', 'warning', 'error'];
    const index = Math.abs(category.split('').reduce((a, b) => a + b.charCodeAt(0), 0)) % colors.length;
    return colors[index] as any;
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('sv-SE', {
      style: 'currency',
      currency: 'SEK',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(amount);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const handleViewDetails = (fund: FundInfo) => {
    setSelectedFund(fund);
    setDetailsOpen(true);
  };

  const handleCloseDetails = () => {
    setDetailsOpen(false);
    setSelectedFund(null);
  };

  if (isLoading) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Fund Explorer
        </Typography>
        <Grid container spacing={3}>
          {Array.from({ length: 6 }).map((_, index) => (
            <Grid item xs={12} md={6} lg={4} key={index}>
              <LoadingCard />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Fund Explorer
        </Typography>
        <ErrorCard 
          title="Fund Data" 
          error={error} 
          onRetry={refetch}
          height="auto"
        />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Fund Explorer
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Explore our curated selection of Swedish investment funds
      </Typography>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                placeholder="Search funds..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                }}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Category</InputLabel>
                <Select
                  value={categoryFilter}
                  label="Category"
                  onChange={(e) => setCategoryFilter(e.target.value)}
                >
                  <MenuItem value="all">All Categories</MenuItem>
                  {categories.map((category) => (
                    <MenuItem key={category} value={category}>
                      {category}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Risk Level</InputLabel>
                <Select
                  value={riskFilter}
                  label="Risk Level"
                  onChange={(e) => setRiskFilter(e.target.value)}
                >
                  <MenuItem value="all">All Risk Levels</MenuItem>
                  {riskLevels.map((risk) => (
                    <MenuItem key={risk} value={risk}>
                      {risk}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={2}>
              <Typography variant="body2" color="text.secondary">
                {filteredFunds.length} of {funds?.length || 0} funds
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Fund Grid */}
      <Grid container spacing={3}>
        {filteredFunds.map((fund) => (
          <Grid item xs={12} md={6} lg={4} key={fund.id}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardContent sx={{ flexGrow: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2 }}>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" gutterBottom>
                      {fund.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {fund.id}
                    </Typography>
                  </Box>
                  <IconButton
                    size="small"
                    onClick={() => handleViewDetails(fund)}
                    aria-label="View details"
                  >
                    <InfoIcon />
                  </IconButton>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Chip
                    label={fund.category}
                    color={getCategoryColor(fund.category)}
                    size="small"
                    sx={{ mr: 1, mb: 1 }}
                  />
                  <Chip
                    label={`${fund.risk_level} Risk`}
                    color={getRiskColor(fund.risk_level)}
                    size="small"
                    sx={{ mb: 1 }}
                  />
                </Box>

                <Typography variant="body2" sx={{ mb: 2 }}>
                  {fund.description}
                </Typography>

                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Expense Ratio:
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {formatPercentage(fund.expense_ratio)}
                  </Typography>
                </Box>

                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Currency:
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {fund.currency}
                  </Typography>
                </Box>

                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Inception:
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {new Date(fund.inception_date).getFullYear()}
                  </Typography>
                </Box>
              </CardContent>
              
              <Box sx={{ p: 2, pt: 0 }}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<TrendingUpIcon />}
                  onClick={() => handleViewDetails(fund)}
                >
                  View Details
                </Button>
              </Box>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Fund Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={handleCloseDetails}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {selectedFund?.name}
        </DialogTitle>
        <DialogContent>
          {selectedFund && (
            <Box>
              {/* Basic Info */}
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="h6">Fund Information</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Fund ID
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {selectedFund.id}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Category
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {selectedFund.category}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Risk Level
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {selectedFund.risk_level}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Currency
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {selectedFund.currency}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Expense Ratio
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {formatPercentage(selectedFund.expense_ratio)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Inception Date
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {new Date(selectedFund.inception_date).toLocaleDateString()}
                      </Typography>
                    </Grid>
                  </Grid>
                  
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Description
                    </Typography>
                    <Typography variant="body1">
                      {selectedFund.description}
                    </Typography>
                  </Box>
                </AccordionDetails>
              </Accordion>

              {/* Performance Data */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
                    <ChartIcon sx={{ mr: 1 }} />
                    Performance Data
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  {performanceLoading ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                      <Typography>Loading performance data...</Typography>
                    </Box>
                  ) : performance ? (
                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Metric</TableCell>
                            <TableCell align="right">Value</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(performance).map(([key, value]) => (
                            <TableRow key={key}>
                              <TableCell>{key.replace(/_/g, ' ').toUpperCase()}</TableCell>
                              <TableCell align="right">
                                {typeof value === 'number' 
                                  ? value > 1 
                                    ? formatCurrency(value)
                                    : formatPercentage(value)
                                  : String(value)
                                }
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  ) : (
                    <Typography>No performance data available</Typography>
                  )}
                </AccordionDetails>
              </Accordion>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDetails}>
            Close
          </Button>
          <Button
            variant="contained"
            onClick={() => {
              // Add to portfolio logic would go here
              handleCloseDetails();
            }}
          >
            Add to Portfolio
          </Button>
        </DialogActions>
      </Dialog>

      {/* Empty State */}
      {filteredFunds.length === 0 && !isLoading && (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 6 }}>
            <FilterIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No funds found
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Try adjusting your search criteria or filters
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default FundExplorer;