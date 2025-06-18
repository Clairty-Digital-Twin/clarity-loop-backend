import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
} from '@mui/material';
import {
  Visibility,
  Search,
  Error,
  CheckCircle,
} from '@mui/icons-material';

function Traces({ socket, realTimeData }) {
  const [traces, setTraces] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock trace data
    const mockTraces = Array.from({ length: 20 }, (_, i) => ({
      id: `trace_${i.toString().padStart(4, '0')}`,
      operation: ['GET /api/v1/health-data', 'POST /api/v1/pat/analyze', 'GET /api/v1/insights'][i % 3],
      duration: (Math.random() * 2000 + 100).toFixed(2),
      spans: Math.floor(Math.random() * 10) + 3,
      timestamp: new Date(Date.now() - i * 60000).toISOString(),
      status: Math.random() > 0.8 ? 'error' : 'success',
      service: 'clarity-backend',
    }));
    
    setTraces(mockTraces);
    setLoading(false);
  }, []);

  const filteredTraces = traces.filter(trace => {
    const matchesSearch = trace.operation.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         trace.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || trace.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const getStatusColor = (status) => {
    return status === 'error' ? 'error' : 'success';
  };

  const getStatusIcon = (status) => {
    return status === 'error' ? <Error /> : <CheckCircle />;
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        🔍 Distributed Traces
      </Typography>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <TextField
              label="Search traces"
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
              <InputLabel>Status</InputLabel>
              <Select
                value={statusFilter}
                label="Status"
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="success">Success</MenuItem>
                <MenuItem value="error">Error</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </CardContent>
      </Card>

      {/* Integration Notice */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography>
          Traces are collected via OpenTelemetry and exported to Jaeger. 
          View detailed traces in <strong>Jaeger UI</strong> at http://localhost:16686
        </Typography>
      </Alert>

      {/* Traces Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recent Traces ({filteredTraces.length})
          </Typography>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Trace ID</TableCell>
                  <TableCell>Operation</TableCell>
                  <TableCell>Duration</TableCell>
                  <TableCell>Spans</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredTraces.map((trace) => (
                  <TableRow key={trace.id} hover>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {trace.id}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {trace.operation}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography 
                        variant="body2" 
                        color={parseFloat(trace.duration) > 1000 ? 'error' : 'text.primary'}
                      >
                        {trace.duration}ms
                      </Typography>
                    </TableCell>
                    <TableCell>{trace.spans}</TableCell>
                    <TableCell>
                      <Chip
                        icon={getStatusIcon(trace.status)}
                        label={trace.status}
                        color={getStatusColor(trace.status)}
                        size="small"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {new Date(trace.timestamp).toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <IconButton 
                        size="small" 
                        onClick={() => window.open(`http://localhost:16686/trace/${trace.id}`, '_blank')}
                      >
                        <Visibility />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
}

export default Traces;