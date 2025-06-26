import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  Alert,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RestartIcon,
  Storage as DatabaseIcon,
  Code as MCPIcon,
  Api as APIIcon,
  Timeline as DataIcon,
  Settings as SettingsIcon,
  Delete as DeleteIcon,
  Backup as BackupIcon,
  BugReport as TestIcon,
  ViewList as LogsIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { systemApiService, MCPTool } from '../services/systemApi';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div role="tabpanel" hidden={value !== index}>
    {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
  </div>
);

const SystemControl: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [testDialogOpen, setTestDialogOpen] = useState(false);
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null);
  const [testParams, setTestParams] = useState('{}');
  const [logFilter, setLogFilter] = useState({ component: '', level: '' });
  
  const queryClient = useQueryClient();

  // System Status Query
  const { data: systemStatus, isLoading: statusLoading } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: systemApiService.getSystemStatus,
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // MCP Tools Query
  const { data: mcpTools, isLoading: toolsLoading } = useQuery({
    queryKey: ['mcpTools'],
    queryFn: systemApiService.getMCPTools,
    refetchInterval: 10000,
  });

  // Data Collection Jobs Query
  const { data: dataJobs, isLoading: jobsLoading } = useQuery({
    queryKey: ['dataJobs'],
    queryFn: systemApiService.getDataCollectionJobs,
    refetchInterval: 3000,
  });

  // System Logs Query
  const { data: systemLogs, isLoading: logsLoading } = useQuery({
    queryKey: ['systemLogs', logFilter],
    queryFn: () => systemApiService.getSystemLogs(
      logFilter.component || undefined,
      logFilter.level || undefined,
      100
    ),
    refetchInterval: 5000,
  });

  // Backend Control Mutations
  const startBackendMutation = useMutation({
    mutationFn: systemApiService.startBackendAPI,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['systemStatus'] }),
  });

  const stopBackendMutation = useMutation({
    mutationFn: systemApiService.stopBackendAPI,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['systemStatus'] }),
  });

  const restartBackendMutation = useMutation({
    mutationFn: systemApiService.restartBackendAPI,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['systemStatus'] }),
  });

  // MCP Control Mutations
  const startMCPMutation = useMutation({
    mutationFn: systemApiService.startMCPServer,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['systemStatus'] }),
  });

  const stopMCPMutation = useMutation({
    mutationFn: systemApiService.stopMCPServer,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['systemStatus'] }),
  });

  const restartMCPMutation = useMutation({
    mutationFn: systemApiService.restartMCPServer,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['systemStatus'] }),
  });

  // Data Collection Mutations
  const startDataCollectionMutation = useMutation({
    mutationFn: (sources: string[]) => systemApiService.startDataCollection(sources),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['dataJobs'] }),
  });

  // Tool Test Mutation
  const testToolMutation = useMutation({
    mutationFn: ({ toolName, params }: { toolName: string; params: any }) => 
      systemApiService.testMCPTool(toolName, params),
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
      case 'connected':
      case 'available':
      case 'completed':
        return 'success';
      case 'stopped':
      case 'disconnected':
      case 'pending':
        return 'warning';
      case 'error':
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
      case 'connected':
      case 'available':
      case 'completed':
        return <SuccessIcon color="success" />;
      case 'error':
      case 'failed':
        return <ErrorIcon color="error" />;
      default:
        return <WarningIcon color="warning" />;
    }
  };

  const handleTestTool = (tool: MCPTool) => {
    setSelectedTool(tool);
    setTestParams('{}');
    setTestDialogOpen(true);
  };

  const executeToolTest = () => {
    if (!selectedTool) return;
    
    try {
      const params = JSON.parse(testParams);
      testToolMutation.mutate({ toolName: selectedTool.name, params });
    } catch (error) {
      alert('Invalid JSON parameters');
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Control Center
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Monitor and control the entire Investment MCP system
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
          <Tab label="System Status" icon={<SettingsIcon />} iconPosition="start" />
          <Tab label="MCP Tools" icon={<MCPIcon />} iconPosition="start" />
          <Tab label="Data Collection" icon={<DataIcon />} iconPosition="start" />
          <Tab label="System Logs" icon={<LogsIcon />} iconPosition="start" />
          <Tab label="Database" icon={<DatabaseIcon />} iconPosition="start" />
        </Tabs>
      </Box>

      {/* System Status Tab */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          {/* Backend API Status */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <APIIcon sx={{ mr: 1 }} />
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    Backend API
                  </Typography>
                  {systemStatus && getStatusIcon(systemStatus.backend_api.status)}
                </Box>
                
                {systemStatus && (
                  <>
                    <Box sx={{ mb: 2 }}>
                      <Chip
                        label={systemStatus.backend_api.status.toUpperCase()}
                        color={getStatusColor(systemStatus.backend_api.status)}
                        sx={{ mr: 1 }}
                      />
                      <Typography variant="body2" color="text.secondary">
                        Port: {systemStatus.backend_api.port}
                      </Typography>
                      {systemStatus.backend_api.uptime && (
                        <Typography variant="body2" color="text.secondary">
                          Uptime: {systemStatus.backend_api.uptime}
                        </Typography>
                      )}
                    </Box>

                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<StartIcon />}
                        onClick={() => startBackendMutation.mutate()}
                        disabled={systemStatus.backend_api.status === 'running' || startBackendMutation.isPending}
                      >
                        Start
                      </Button>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<StopIcon />}
                        onClick={() => stopBackendMutation.mutate()}
                        disabled={systemStatus.backend_api.status === 'stopped' || stopBackendMutation.isPending}
                      >
                        Stop
                      </Button>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<RestartIcon />}
                        onClick={() => restartBackendMutation.mutate()}
                        disabled={restartBackendMutation.isPending}
                      >
                        Restart
                      </Button>
                    </Box>
                  </>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* MCP Server Status */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <MCPIcon sx={{ mr: 1 }} />
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    MCP Server
                  </Typography>
                  {systemStatus && getStatusIcon(systemStatus.mcp_server.status)}
                </Box>
                
                {systemStatus && (
                  <>
                    <Box sx={{ mb: 2 }}>
                      <Chip
                        label={systemStatus.mcp_server.status.toUpperCase()}
                        color={getStatusColor(systemStatus.mcp_server.status)}
                        sx={{ mr: 1 }}
                      />
                      <Typography variant="body2" color="text.secondary">
                        Port: {systemStatus.mcp_server.port}
                      </Typography>
                      {systemStatus.mcp_server.tools_available !== undefined && (
                        <Typography variant="body2" color="text.secondary">
                          Tools: {systemStatus.mcp_server.tools_available} available
                        </Typography>
                      )}
                    </Box>

                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<StartIcon />}
                        onClick={() => startMCPMutation.mutate()}
                        disabled={systemStatus.mcp_server.status === 'running' || startMCPMutation.isPending}
                      >
                        Start
                      </Button>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<StopIcon />}
                        onClick={() => stopMCPMutation.mutate()}
                        disabled={systemStatus.mcp_server.status === 'stopped' || stopMCPMutation.isPending}
                      >
                        Stop
                      </Button>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<RestartIcon />}
                        onClick={() => restartMCPMutation.mutate()}
                        disabled={restartMCPMutation.isPending}
                      >
                        Restart
                      </Button>
                    </Box>
                  </>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Database Status */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <DatabaseIcon sx={{ mr: 1 }} />
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    Database
                  </Typography>
                  {systemStatus && getStatusIcon(systemStatus.database.status)}
                </Box>
                
                {systemStatus && (
                  <Box>
                    <Chip
                      label={systemStatus.database.status.toUpperCase()}
                      color={getStatusColor(systemStatus.database.status)}
                      sx={{ mb: 1 }}
                    />
                    <Typography variant="body2" color="text.secondary">
                      Path: {systemStatus.database.path}
                    </Typography>
                    {systemStatus.database.size && (
                      <Typography variant="body2" color="text.secondary">
                        Size: {systemStatus.database.size}
                      </Typography>
                    )}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Data Collection Status */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <DataIcon sx={{ mr: 1 }} />
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    Data Collection
                  </Typography>
                  {systemStatus && getStatusIcon(systemStatus.data_collectors.status)}
                </Box>
                
                {systemStatus && (
                  <>
                    <Box sx={{ mb: 2 }}>
                      <Chip
                        label={systemStatus.data_collectors.status.toUpperCase()}
                        color={getStatusColor(systemStatus.data_collectors.status)}
                        sx={{ mb: 1 }}
                      />
                      {systemStatus.data_collectors.last_run && (
                        <Typography variant="body2" color="text.secondary">
                          Last run: {systemStatus.data_collectors.last_run}
                        </Typography>
                      )}
                      {systemStatus.data_collectors.collected_records !== undefined && (
                        <Typography variant="body2" color="text.secondary">
                          Records: {systemStatus.data_collectors.collected_records}
                        </Typography>
                      )}
                    </Box>

                    <Button
                      variant="contained"
                      size="small"
                      startIcon={<StartIcon />}
                      onClick={() => startDataCollectionMutation.mutate(['all'])}
                      disabled={startDataCollectionMutation.isPending}
                    >
                      Start Collection
                    </Button>
                  </>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* MCP Tools Tab */}
      <TabPanel value={tabValue} index={1}>
        <Box sx={{ mb: 2 }}>
          <Typography variant="h6" gutterBottom>
            Available MCP Tools
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Test and monitor MCP server tools
          </Typography>
        </Box>

        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Tool Name</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Usage Count</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {mcpTools?.map((tool) => (
                <TableRow key={tool.name}>
                  <TableCell>
                    <Typography variant="body2" fontWeight="medium">
                      {tool.name}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {tool.description}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={tool.status}
                      color={getStatusColor(tool.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {tool.usage_count || 0}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Button
                      size="small"
                      startIcon={<TestIcon />}
                      onClick={() => handleTestTool(tool)}
                      disabled={tool.status !== 'available'}
                    >
                      Test
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* Data Collection Tab */}
      <TabPanel value={tabValue} index={2}>
        <Box sx={{ mb: 2 }}>
          <Typography variant="h6" gutterBottom>
            Data Collection Jobs
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <Button
              variant="contained"
              startIcon={<StartIcon />}
              onClick={() => startDataCollectionMutation.mutate(['yfinance'])}
            >
              Collect Fund Data
            </Button>
            <Button
              variant="contained"
              startIcon={<StartIcon />}
              onClick={() => startDataCollectionMutation.mutate(['riksbank'])}
            >
              Collect Economic Data
            </Button>
            <Button
              variant="contained"
              startIcon={<StartIcon />}
              onClick={() => startDataCollectionMutation.mutate(['all'])}
            >
              Collect All Data
            </Button>
          </Box>
        </Box>

        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Job ID</TableCell>
                <TableCell>Source</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Progress</TableCell>
                <TableCell>Records</TableCell>
                <TableCell>Started</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {dataJobs?.map((job) => (
                <TableRow key={job.id}>
                  <TableCell>
                    <Typography variant="body2" fontFamily="monospace">
                      {job.id}
                    </Typography>
                  </TableCell>
                  <TableCell>{job.source}</TableCell>
                  <TableCell>
                    <Chip
                      label={job.status}
                      color={getStatusColor(job.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={job.progress}
                        sx={{ flexGrow: 1, height: 6 }}
                      />
                      <Typography variant="body2">
                        {job.progress}%
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>{job.records_collected || 0}</TableCell>
                  <TableCell>
                    {job.started_at && new Date(job.started_at).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    {job.status === 'running' && (
                      <IconButton
                        size="small"
                        onClick={() => systemApiService.cancelDataCollectionJob(job.id)}
                      >
                        <StopIcon />
                      </IconButton>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* System Logs Tab */}
      <TabPanel value={tabValue} index={3}>
        <Box sx={{ mb: 2 }}>
          <Typography variant="h6" gutterBottom>
            System Logs
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Component</InputLabel>
              <Select
                value={logFilter.component}
                label="Component"
                onChange={(e) => setLogFilter({ ...logFilter, component: e.target.value })}
              >
                <MenuItem value="">All</MenuItem>
                <MenuItem value="backend">Backend</MenuItem>
                <MenuItem value="mcp">MCP Server</MenuItem>
                <MenuItem value="dashboard">Dashboard</MenuItem>
                <MenuItem value="data">Data Collection</MenuItem>
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Level</InputLabel>
              <Select
                value={logFilter.level}
                label="Level"
                onChange={(e) => setLogFilter({ ...logFilter, level: e.target.value })}
              >
                <MenuItem value="">All</MenuItem>
                <MenuItem value="error">Error</MenuItem>
                <MenuItem value="warning">Warning</MenuItem>
                <MenuItem value="info">Info</MenuItem>
                <MenuItem value="debug">Debug</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Box>

        <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell>Timestamp</TableCell>
                <TableCell>Level</TableCell>
                <TableCell>Component</TableCell>
                <TableCell>Message</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {systemLogs?.map((log, index) => (
                <TableRow key={index}>
                  <TableCell>
                    <Typography variant="caption" fontFamily="monospace">
                      {new Date(log.timestamp).toLocaleString()}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={log.level}
                      color={getStatusColor(log.level === 'info' ? 'available' : log.level)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{log.component}</TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {log.message}
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* Database Tab */}
      <TabPanel value={tabValue} index={4}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Database Operations
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Button
                    variant="contained"
                    startIcon={<BackupIcon />}
                    onClick={() => systemApiService.backupDatabase()}
                  >
                    Backup Database
                  </Button>
                  <Button
                    variant="outlined"
                    color="error"
                    startIcon={<DeleteIcon />}
                    onClick={() => {
                      if (window.confirm('Are you sure you want to reset the database? This will delete all data.')) {
                        systemApiService.resetDatabase();
                      }
                    }}
                  >
                    Reset Database
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Tool Test Dialog */}
      <Dialog open={testDialogOpen} onClose={() => setTestDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Test MCP Tool: {selectedTool?.name}
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {selectedTool?.description}
          </Typography>
          <TextField
            fullWidth
            multiline
            rows={6}
            label="Parameters (JSON)"
            value={testParams}
            onChange={(e) => setTestParams(e.target.value)}
            placeholder='{"example": "value"}'
            sx={{ mb: 2 }}
          />
          {testToolMutation.data && (
            <Alert severity={testToolMutation.data.success ? 'success' : 'error'} sx={{ mb: 2 }}>
              {testToolMutation.data.success ? (
                <pre>{JSON.stringify(testToolMutation.data.result, null, 2)}</pre>
              ) : (
                testToolMutation.data.error
              )}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTestDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={executeToolTest}
            disabled={testToolMutation.isPending}
          >
            Test Tool
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SystemControl;