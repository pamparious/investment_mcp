import React from 'react';
import { Card, CardContent, Skeleton, Box } from '@mui/material';

interface LoadingCardProps {
  height?: number | string;
  title?: boolean;
  lines?: number;
}

const LoadingCard: React.FC<LoadingCardProps> = ({ 
  height = 200, 
  title = true, 
  lines = 3 
}) => {
  return (
    <Card sx={{ height }}>
      <CardContent>
        {title && (
          <Skeleton variant="text" sx={{ fontSize: '1.5rem', mb: 2 }} width="60%" />
        )}
        {Array.from({ length: lines }).map((_, index) => (
          <Skeleton
            key={index}
            variant="text"
            sx={{ fontSize: '1rem', mb: 1 }}
            width={index === lines - 1 ? '80%' : '100%'}
          />
        ))}
        <Box sx={{ mt: 2 }}>
          <Skeleton variant="rectangular" height={60} />
        </Box>
      </CardContent>
    </Card>
  );
};

export default LoadingCard;