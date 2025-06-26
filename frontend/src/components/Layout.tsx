import React, { useState } from 'react';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  IconButton,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  useTheme,
  useMediaQuery,
  Alert,
  Chip,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  TrendingUp as TrendingUpIcon,
  AccountBalance as FundsIcon,
  ShowChart as MarketIcon,
  Settings as SettingsIcon,
  Help as HelpIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Build as SystemIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { apiService, queryKeys } from '../services/api';

const drawerWidth = 240;

interface LayoutProps {
  children: React.ReactNode;
}

interface NavigationItem {
  text: string;
  icon: React.ReactElement;
  path: string;
}

const navigationItems: NavigationItem[] = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
  { text: 'Portfolio Analysis', icon: <TrendingUpIcon />, path: '/portfolio' },
  { text: 'Fund Explorer', icon: <FundsIcon />, path: '/funds' },
  { text: 'Market Data', icon: <MarketIcon />, path: '/market' },
  { text: 'System Control', icon: <SystemIcon />, path: '/system' },
  { text: 'API Tester', icon: <SettingsIcon />, path: '/api-tester' },
];

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const navigate = useNavigate();
  const location = useLocation();

  // API Health Check
  const { data: healthStatus, isLoading: healthLoading, error: healthError } = useQuery({
    queryKey: queryKeys.health,
    queryFn: apiService.getHealth,
    refetchInterval: 30000, // Check every 30 seconds
  });

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleNavigation = (path: string) => {
    navigate(path);
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  const getApiStatusColor = () => {
    if (healthLoading) return 'default';
    if (healthError) return 'error';
    return healthStatus?.status === 'healthy' ? 'success' : 'warning';
  };

  const getApiStatusText = () => {
    if (healthLoading) return 'Checking...';
    if (healthError) return 'Offline';
    return healthStatus?.status === 'healthy' ? 'Online' : 'Issues';
  };

  const drawer = (
    <Box>
      <Toolbar>
        <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 'bold' }}>
          Investment MCP
        </Typography>
      </Toolbar>
      <Divider />
      
      {/* API Status */}
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
          API Status
        </Typography>
        <Chip
          icon={healthError ? <ErrorIcon /> : <CheckIcon />}
          label={getApiStatusText()}
          color={getApiStatusColor()}
          size="small"
          variant="outlined"
        />
      </Box>
      <Divider />

      {/* Navigation */}
      <List>
        {navigationItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => handleNavigation(item.path)}
              sx={{
                '&.Mui-selected': {
                  backgroundColor: theme.palette.primary.main + '20',
                  '&:hover': {
                    backgroundColor: theme.palette.primary.main + '30',
                  },
                },
              }}
            >
              <ListItemIcon
                sx={{
                  color: location.pathname === item.path 
                    ? theme.palette.primary.main 
                    : 'inherit'
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.text}
                sx={{
                  '& .MuiListItemText-primary': {
                    fontWeight: location.pathname === item.path ? 600 : 400,
                    color: location.pathname === item.path 
                      ? theme.palette.primary.main 
                      : 'inherit'
                  }
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />

      {/* Secondary Navigation */}
      <List>
        <ListItem disablePadding>
          <ListItemButton onClick={() => window.open('http://localhost:8000/docs', '_blank')}>
            <ListItemIcon>
              <HelpIcon />
            </ListItemIcon>
            <ListItemText primary="API Docs" />
          </ListItemButton>
        </ListItem>
        <ListItem disablePadding>
          <ListItemButton>
            <ListItemIcon>
              <SettingsIcon />
            </ListItemIcon>
            <ListItemText primary="Settings" />
          </ListItemButton>
        </ListItem>
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          backgroundColor: 'white',
          color: 'text.primary',
          boxShadow: 1,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Swedish Investment Dashboard
          </Typography>
          
          {/* Version Info */}
          {healthStatus && (
            <Chip
              label={`v${healthStatus.version}`}
              size="small"
              variant="outlined"
              sx={{ ml: 2 }}
            />
          )}
        </Toolbar>
      </AppBar>

      {/* Drawer */}
      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
            },
          }}
        >
          {drawer}
        </Drawer>
        
        {/* Desktop drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          backgroundColor: '#f5f5f5',
        }}
      >
        <Toolbar />
        
        {/* Connection Error Alert */}
        {healthError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            Unable to connect to the Investment API. Please ensure the backend server is running on port 8000.
          </Alert>
        )}
        
        {children}
      </Box>
    </Box>
  );
};

export default Layout;