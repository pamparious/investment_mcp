/**
 * React Error Boundary component for graceful error handling
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Alert,
  Collapse,
  IconButton,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandIcon,
  BugReport as BugIcon,
} from '@mui/icons-material';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  showDetails: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error,
      errorInfo,
    });

    // Log error to console for development
    console.error('Error caught by boundary:', error, errorInfo);

    // Call custom error handler if provided
    this.props.onError?.(error, errorInfo);

    // In production, you might want to log to an error reporting service
    if (process.env.NODE_ENV === 'production') {
      // Example: logErrorToService(error, errorInfo);
    }
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
    });
  };

  handleReload = () => {
    window.location.reload();
  };

  toggleDetails = () => {
    this.setState(prev => ({ showDetails: !prev.showDetails }));
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback component if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '400px',
            p: 3,
          }}
        >
          <Card sx={{ maxWidth: 600, width: '100%' }}>
            <CardContent sx={{ textAlign: 'center', p: 3 }}>
              <ErrorIcon
                sx={{ fontSize: 64, color: 'error.main', mb: 2 }}
              />
              
              <Typography variant="h5" gutterBottom>
                Something went wrong
              </Typography>
              
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                We're sorry, but something unexpected happened. Please try refreshing the page or contact support if the problem persists.
              </Typography>

              <Alert severity="error" sx={{ mb: 3, textAlign: 'left' }}>
                <Typography variant="body2">
                  <strong>Error:</strong> {this.state.error?.message || 'Unknown error'}
                </Typography>
              </Alert>

              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mb: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<RefreshIcon />}
                  onClick={this.handleRetry}
                >
                  Try Again
                </Button>
                
                <Button
                  variant="outlined"
                  onClick={this.handleReload}
                >
                  Reload Page
                </Button>
              </Box>

              {/* Development error details */}
              {process.env.NODE_ENV === 'development' && (
                <Box sx={{ mt: 3 }}>
                  <Button
                    variant="text"
                    startIcon={<BugIcon />}
                    endIcon={<ExpandIcon sx={{ 
                      transform: this.state.showDetails ? 'rotate(180deg)' : 'rotate(0deg)',
                      transition: 'transform 0.2s',
                    }} />}
                    onClick={this.toggleDetails}
                    size="small"
                  >
                    {this.state.showDetails ? 'Hide' : 'Show'} Error Details
                  </Button>
                  
                  <Collapse in={this.state.showDetails}>
                    <Box
                      sx={{
                        mt: 2,
                        p: 2,
                        backgroundColor: 'grey.100',
                        borderRadius: 1,
                        textAlign: 'left',
                      }}
                    >
                      <Typography variant="caption" component="div" sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap', fontSize: '0.75rem' }}>
                        <strong>Stack Trace:</strong>
                        {'\n'}
                        {this.state.error?.stack}
                        {'\n\n'}
                        <strong>Component Stack:</strong>
                        {'\n'}
                        {this.state.errorInfo?.componentStack}
                      </Typography>
                    </Box>
                  </Collapse>
                </Box>
              )}
            </CardContent>
          </Card>
        </Box>
      );
    }

    return this.props.children;
  }
}

// Higher-order component for wrapping components with error boundary
export const withErrorBoundary = <P extends object>(
  Component: React.ComponentType<P>,
  fallback?: ReactNode,
  onError?: (error: Error, errorInfo: ErrorInfo) => void
) => {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary fallback={fallback} onError={onError}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  
  return WrappedComponent;
};

// Simple error fallback component for specific sections
export const ErrorFallback: React.FC<{
  error?: Error;
  onRetry?: () => void;
  title?: string;
}> = ({ error, onRetry, title = 'Error' }) => (
  <Alert 
    severity="error" 
    action={
      onRetry && (
        <IconButton color="inherit" size="small" onClick={onRetry}>
          <RefreshIcon fontSize="small" />
        </IconButton>
      )
    }
  >
    <Typography variant="subtitle2">{title}</Typography>
    {error && (
      <Typography variant="body2" sx={{ mt: 1 }}>
        {error.message}
      </Typography>
    )}
  </Alert>
);

export default ErrorBoundary;