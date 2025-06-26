import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Chip,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Send as SendIcon,
  Code as CodeIcon,
} from '@mui/icons-material';
import { useMutation } from '@tanstack/react-query';
import axios from 'axios';

interface APIEndpoint {
  name: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  description: string;
  parameters?: { name: string; type: string; required: boolean; description: string }[];
  body_example?: string;
}

interface APITest {
  endpoint: APIEndpoint;
  parameters: Record<string, any>;
  body: string;
  response?: any;
  status?: number;
  error?: string;
  timestamp?: string;
}

const API_ENDPOINTS: APIEndpoint[] = [
  {
    name: 'Health Check',
    method: 'GET',
    path: '/api/v1/health',
    description: 'Check API health status',
  },
  {
    name: 'System Status',
    method: 'GET',
    path: '/',
    description: 'Get system information',
  },
  {
    name: 'Get All Funds',
    method: 'GET',
    path: '/api/v1/funds',
    description: 'Retrieve all available Swedish funds',
  },
  {
    name: 'Portfolio Analysis',
    method: 'POST',
    path: '/api/v1/portfolio/analysis',
    description: 'Analyze portfolio and get AI recommendations',
    body_example: JSON.stringify({
      risk_tolerance: 'medium',
      investment_horizon: 'medium',
      investment_amount: 100000
    }, null, 2),
  },
  {
    name: 'Market Sentiment',
    method: 'GET',
    path: '/api/v1/market/sentiment',
    description: 'Get current market sentiment analysis',
  },
  {
    name: 'Economic Indicators',
    method: 'GET',
    path: '/api/v1/economic/sweden/indicators',
    description: 'Get Swedish economic indicators',
  },
  {
    name: 'Fund Performance',
    method: 'GET',
    path: '/api/v1/funds/{fund_id}/performance',
    description: 'Get performance data for a specific fund',
    parameters: [
      { name: 'fund_id', type: 'string', required: true, description: 'Fund identifier' },
      { name: 'period', type: 'string', required: false, description: 'Time period (1y, 6m, 3m)' },
    ],
  },
  {
    name: 'AI Investment Recommendation',
    method: 'POST',
    path: '/api/v1/ai/investment-recommendation',
    description: 'Get AI-powered investment recommendations',
    body_example: JSON.stringify({
      risk_profile: 'moderate',
      investment_goal: 'growth',
      time_horizon: 'long_term',
      amount: 500000
    }, null, 2),
  },
];

const APITester: React.FC = () => {
  const [selectedEndpoint, setSelectedEndpoint] = useState<APIEndpoint>(API_ENDPOINTS[0]);
  const [apiBase, setApiBase] = useState('http://localhost:8000');
  const [apiKey, setApiKey] = useState('dev-key-123');
  const [requestBody, setRequestBody] = useState('');
  const [pathParams, setPathParams] = useState<Record<string, string>>({});
  const [queryParams, setQueryParams] = useState<Record<string, string>>({});
  const [testHistory, setTestHistory] = useState<APITest[]>([]);
  const [tabValue, setTabValue] = useState(0);

  const apiTestMutation = useMutation({
    mutationFn: async (test: Omit<APITest, 'response' | 'status' | 'error' | 'timestamp'>) => {
      let url = `${apiBase}${test.endpoint.path}`;
      
      // Replace path parameters
      Object.entries(pathParams).forEach(([key, value]) => {
        url = url.replace(`{${key}}`, value);
      });

      // Add query parameters
      const params = new URLSearchParams();
      Object.entries(queryParams).forEach(([key, value]) => {
        if (value) params.append(key, value);
      });
      if (params.toString()) {
        url += `?${params.toString()}`;
      }

      const config: any = {
        method: test.endpoint.method,
        url,
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey,
        },
      };

      if (test.endpoint.method !== 'GET' && test.body) {
        config.data = JSON.parse(test.body);
      }

      const response = await axios(config);
      return response;
    },
    onSuccess: (response, variables) => {
      const test: APITest = {
        ...variables,
        response: response.data,
        status: response.status,
        timestamp: new Date().toISOString(),
      };
      setTestHistory(prev => [test, ...prev.slice(0, 19)]); // Keep last 20 tests
    },
    onError: (error: any, variables) => {
      const test: APITest = {
        ...variables,
        status: error.response?.status || 0,
        error: error.response?.data?.error || error.message,
        timestamp: new Date().toISOString(),
      };
      setTestHistory(prev => [test, ...prev.slice(0, 19)]);
    },
  });

  const handleEndpointChange = (endpoint: APIEndpoint) => {
    setSelectedEndpoint(endpoint);
    setRequestBody(endpoint.body_example || '');
    setPathParams({});
    setQueryParams({});
  };

  const handleTest = () => {
    apiTestMutation.mutate({
      endpoint: selectedEndpoint,
      parameters: { ...pathParams, ...queryParams },
      body: requestBody,
    });
  };

  const getStatusColor = (status?: number) => {
    if (!status) return 'error';
    if (status >= 200 && status < 300) return 'success';
    if (status >= 400 && status < 500) return 'warning';
    return 'error';
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        API Tester
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Test and debug Investment MCP API endpoints
      </Typography>

      <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)} sx={{ mb: 3 }}>
        <Tab label="API Testing" />
        <Tab label="Test History" />
        <Tab label="API Documentation" />
      </Tabs>

      {/* API Testing Tab */}
      {tabValue === 0 && (
        <Grid container spacing={3}>
          {/* Configuration Panel */}
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Configuration
                </Typography>
                
                <TextField
                  fullWidth
                  label="API Base URL"
                  value={apiBase}
                  onChange={(e) => setApiBase(e.target.value)}
                  sx={{ mb: 2 }}
                />
                
                <TextField
                  fullWidth
                  label="API Key"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  sx={{ mb: 2 }}
                />
                
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Endpoint</InputLabel>
                  <Select
                    value={selectedEndpoint.name}
                    label="Endpoint"
                    onChange={(e) => {
                      const endpoint = API_ENDPOINTS.find(ep => ep.name === e.target.value);
                      if (endpoint) handleEndpointChange(endpoint);
                    }}
                  >
                    {API_ENDPOINTS.map((endpoint) => (
                      <MenuItem key={endpoint.name} value={endpoint.name}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Chip label={endpoint.method} size="small" />
                          {endpoint.name}
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    {selectedEndpoint.description}
                  </Typography>
                </Alert>

                {/* Path Parameters */}
                {selectedEndpoint.parameters && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Path Parameters
                    </Typography>
                    {selectedEndpoint.parameters.map((param) => (
                      <TextField
                        key={param.name}
                        fullWidth
                        size="small"
                        label={param.name}
                        placeholder={param.description}
                        value={pathParams[param.name] || ''}
                        onChange={(e) => setPathParams({
                          ...pathParams,
                          [param.name]: e.target.value
                        })}
                        required={param.required}
                        sx={{ mb: 1 }}
                      />
                    ))}
                  </Box>
                )}

                {/* Query Parameters */}
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Query Parameters
                  </Typography>
                  <TextField
                    fullWidth
                    size="small"
                    label="period"
                    placeholder="Time period (optional)"
                    value={queryParams.period || ''}
                    onChange={(e) => setQueryParams({
                      ...queryParams,
                      period: e.target.value
                    })}
                    sx={{ mb: 1 }}
                  />
                </Box>

                <Button
                  fullWidth
                  variant="contained"
                  startIcon={<SendIcon />}
                  onClick={handleTest}
                  disabled={apiTestMutation.isPending}
                >
                  {apiTestMutation.isPending ? 'Testing...' : 'Test Endpoint'}
                </Button>
              </CardContent>
            </Card>
          </Grid>

          {/* Request/Response Panel */}
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Request & Response
                </Typography>

                {/* Request Body */}
                {selectedEndpoint.method !== 'GET' && (
                  <Accordion defaultExpanded>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">Request Body</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <TextField
                        fullWidth
                        multiline
                        rows={8}
                        label="JSON Body"
                        value={requestBody}
                        onChange={(e) => setRequestBody(e.target.value)}
                        placeholder="Enter JSON request body"
                        sx={{ fontFamily: 'monospace' }}
                      />
                    </AccordionDetails>
                  </Accordion>
                )}

                {/* Response */}
                {(apiTestMutation.data || apiTestMutation.error) && (
                  <Accordion defaultExpanded>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">Response</Typography>
                      {apiTestMutation.data && (
                        <Chip
                          label={`${apiTestMutation.data.status} ${apiTestMutation.data.statusText}`}
                          color={getStatusColor(apiTestMutation.data.status)}
                          size="small"
                          sx={{ ml: 2 }}
                        />
                      )}
                    </AccordionSummary>
                    <AccordionDetails>
                      <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
                        <pre style={{ margin: 0, fontFamily: 'monospace', fontSize: '0.875rem' }}>
                          {apiTestMutation.data ? 
                            JSON.stringify(apiTestMutation.data.data, null, 2) :
                            JSON.stringify(apiTestMutation.error, null, 2)
                          }
                        </pre>
                      </Paper>
                    </AccordionDetails>
                  </Accordion>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Test History Tab */}
      {tabValue === 1 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Test History
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Timestamp</TableCell>
                    <TableCell>Endpoint</TableCell>
                    <TableCell>Method</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Response</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {testHistory.map((test, index) => (
                    <TableRow key={index}>
                      <TableCell>
                        <Typography variant="caption">
                          {test.timestamp && new Date(test.timestamp).toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>{test.endpoint.name}</TableCell>
                      <TableCell>
                        <Chip label={test.endpoint.method} size="small" />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={test.status || 'Error'}
                          color={getStatusColor(test.status)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Button
                          size="small"
                          startIcon={<CodeIcon />}
                          onClick={() => {
                            console.log('Test Result:', test);
                            alert('Response logged to console');
                          }}
                        >
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* API Documentation Tab */}
      {tabValue === 2 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              API Endpoints Documentation
            </Typography>
            {API_ENDPOINTS.map((endpoint) => (
              <Accordion key={endpoint.name}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Chip label={endpoint.method} size="small" />
                    <Typography variant="subtitle1">{endpoint.name}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {endpoint.path}
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    {endpoint.description}
                  </Typography>
                  {endpoint.parameters && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Parameters:
                      </Typography>
                      <ul>
                        {endpoint.parameters.map((param) => (
                          <li key={param.name}>
                            <strong>{param.name}</strong> ({param.type}) 
                            {param.required && ' *'} - {param.description}
                          </li>
                        ))}
                      </ul>
                    </Box>
                  )}
                  {endpoint.body_example && (
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Example Request Body:
                      </Typography>
                      <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
                        <pre style={{ margin: 0, fontFamily: 'monospace', fontSize: '0.875rem' }}>
                          {endpoint.body_example}
                        </pre>
                      </Paper>
                    </Box>
                  )}
                </AccordionDetails>
              </Accordion>
            ))}
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default APITester;