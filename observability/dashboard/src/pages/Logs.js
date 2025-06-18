import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Paper,
  Chip,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Alert,
} from '@mui/material';
import {
  Search,
  PlayArrow,
  Pause,
  Refresh,
} from '@mui/icons-material';

function Logs({ socket, realTimeData }) {
  const [logs, setLogs] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [levelFilter, setLevelFilter] = useState('all');
  const [streaming, setStreaming] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Mock log data
    const mockLogs = Array.from({ length: 50 }, (_, i) => ({
      id: `log_${i}`,
      timestamp: new Date(Date.now() - i * 5000).toISOString(),
      level: ['INFO', 'DEBUG', 'WARNING', 'ERROR'][Math.floor(Math.random() * 4)],
      logger: ['clarity.api.v1.health_data', 'clarity.ml.pat_service', 'clarity.auth.middleware'][i % 3],
      message: [
        'Processing health data request',
        'ML model inference completed',
        'User authentication successful',
        'Database query executed',
        'Cache miss, fetching from database',
        'PAT analysis request received',
        'WebSocket connection established',
      ][i % 7],
      correlation_id: `corr_${(i % 10).toString().padStart(4, '0')}`,
      extra: {
        user_id: i % 5 === 0 ? `user_${i % 10}` : null,
        duration_ms: Math.floor(Math.random() * 500) + 50,
      },
    }));
    
    setLogs(mockLogs);
  }, []);

  const filteredLogs = logs.filter(log => {
    const matchesSearch = log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         log.logger.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         log.correlation_id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesLevel = levelFilter === 'all' || log.level === levelFilter;
    return matchesSearch && matchesLevel;
  });

  const getLevelColor = (level) => {
    switch (level) {
      case 'ERROR': return 'error';
      case 'WARNING': return 'warning';
      case 'INFO': return 'info';
      case 'DEBUG': return 'default';
      default: return 'default';
    }
  };

  const getLevelBgColor = (level, theme) => {
    switch (level) {
      case 'ERROR': return 'rgba(244, 67, 54, 0.1)';
      case 'WARNING': return 'rgba(255, 152, 0, 0.1)';
      case 'INFO': return 'rgba(33, 150, 243, 0.1)';
      default: return 'transparent';
    }
  };

  const toggleStreaming = () => {
    setStreaming(!streaming);
  };

  const handleRefresh = () => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setLoading(false);
    }, 1000);
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        📝 Application Logs
      </Typography>

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2 }}>
            <TextField
              label="Search logs"
              variant="outlined"
              size="small"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />,
              }}
              sx={{ minWidth: 300 }}
            />
            
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Level</InputLabel>
              <Select
                value={levelFilter}
                label="Level"
                onChange={(e) => setLevelFilter(e.target.value)}
              >
                <MenuItem value="all">All Levels</MenuItem>
                <MenuItem value="ERROR">ERROR</MenuItem>
                <MenuItem value="WARNING">WARNING</MenuItem>
                <MenuItem value="INFO">INFO</MenuItem>
                <MenuItem value="DEBUG">DEBUG</MenuItem>
              </Select>
            </FormControl>

            <Button
              variant={streaming ? "contained" : "outlined"}
              startIcon={streaming ? <Pause /> : <PlayArrow />}
              onClick={toggleStreaming}
              color={streaming ? "error" : "primary"}
            >
              {streaming ? 'Stop' : 'Start'} Stream
            </Button>

            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={handleRefresh}
              disabled={loading}
            >
              Refresh
            </Button>
          </Box>

          {streaming && (
            <Alert severity="info">
              Real-time log streaming is active. New logs will appear automatically.
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Log Integration Notice */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography>
          Logs are aggregated with <strong>Grafana Loki</strong>. 
          View advanced log analysis in Grafana at http://localhost:3000
        </Typography>
      </Alert>

      {/* Logs Display */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Log Entries ({filteredLogs.length})
          </Typography>
          
          <Box sx={{ maxHeight: '70vh', overflow: 'auto' }}>
            {filteredLogs.map((log) => (
              <Paper
                key={log.id}
                sx={{
                  p: 2,
                  mb: 1,
                  backgroundColor: getLevelBgColor(log.level),
                  border: '1px solid',
                  borderColor: 'divider',
                  fontFamily: 'monospace',
                  fontSize: '0.85rem',
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Typography variant="body2" color="text.secondary" sx={{ minWidth: 180 }}>
                    {new Date(log.timestamp).toLocaleString()}
                  </Typography>
                  
                  <Chip
                    label={log.level}
                    color={getLevelColor(log.level)}
                    size="small"
                    sx={{ minWidth: 80, fontWeight: 'bold' }}
                  />
                  
                  <Chip
                    label={log.correlation_id}
                    variant="outlined"
                    size="small"
                    sx={{ fontFamily: 'monospace' }}
                  />
                  
                  <Typography variant="body2" color="text.secondary">
                    {log.logger}
                  </Typography>
                </Box>
                
                <Typography variant="body1" sx={{ mb: 1 }}>
                  {log.message}
                </Typography>
                
                {log.extra && Object.keys(log.extra).length > 0 && (
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {Object.entries(log.extra).map(([key, value]) => (
                      value && (
                        <Chip
                          key={key}
                          label={`${key}: ${value}`}
                          variant="outlined"
                          size="small"
                          color="secondary"
                        />
                      )
                    ))}
                  </Box>
                )}
              </Paper>
            ))}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}

export default Logs;