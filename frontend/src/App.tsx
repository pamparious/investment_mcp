import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

import { queryClient } from './services/queryClient';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import PortfolioAnalysis from './pages/PortfolioAnalysis';
import FundExplorer from './pages/FundExplorer';
import MarketData from './pages/MarketData';
import SystemControl from './pages/SystemControl';
import APITester from './pages/APITester';

// Create Material-UI theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: [
      'Roboto',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
  },
  shape: {
    borderRadius: 8,
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/portfolio" element={<PortfolioAnalysis />} />
              <Route path="/funds" element={<FundExplorer />} />
              <Route path="/market" element={<MarketData />} />
              <Route path="/system" element={<SystemControl />} />
              <Route path="/api-tester" element={<APITester />} />
            </Routes>
          </Layout>
        </Router>
      </ThemeProvider>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
