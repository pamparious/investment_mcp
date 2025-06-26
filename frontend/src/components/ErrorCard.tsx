import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Alert, 
  Button, 
  Box 
} from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';

interface ErrorCardProps {
  title: string;
  error: Error | any;
  onRetry?: () => void;
  height?: number | string;
}

const ErrorCard: React.FC<ErrorCardProps> = ({ 
  title, 
  error, 
  onRetry, 
  height = 200 
}) => {
  const getErrorMessage = (error: any): string => {
    if (error?.response?.data?.error) {
      return error.response.data.error;
    }
    if (error?.response?.data?.message) {
      return error.response.data.message;
    }
    if (error?.message) {
      return error.message;
    }
    return 'An unexpected error occurred';
  };

  return (
    <Card sx={{ height }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        
        <Alert severity="error" sx={{ mb: 2 }}>
          {getErrorMessage(error)}
        </Alert>
        
        {onRetry && (
          <Box sx={{ display: 'flex', justifyContent: 'center' }}>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={onRetry}
              size="small"
            >
              Retry
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ErrorCard;