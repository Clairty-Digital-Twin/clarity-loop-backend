import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  LinearProgress,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  Computer,
  Storage,
  Memory,
  Speed,
  Cloud,
  Security,
} from '@mui/icons-material';

function SystemHealth({ socket, realTimeData }) {
  const [healthData, setHealthData] = useState({});
  const [services, setServices] = useState([]);
  const [systemMetrics, setSystemMetrics] = useState({});

  useEffect(() => {
    // Mock health data
    setHealthData({
      status: 'healthy',
      timestamp: Date.now(),
      components: {
        api: 'healthy',
        database: 'healthy',
        redis: 'healthy',
        ml_models: 'degraded',
        authentication: 'healthy',
        storage: 'healthy',
      },
      metrics: {
        cpu_usage: 35.2,
        memory_usage: 768000000, // bytes
        disk_usage: 45.8,
        active_connections: 24,
        response_time: 145,
        uptime: 7200, // seconds
      },
    });

    // Mock services data
    setServices([
      {
        name: 'Clarity Backend API',
        status: 'healthy',
        url: 'http://localhost:8000',
        version: '0.2.0',
        uptime: '2h 15m',
        lastCheck: new Date().toISOString(),
      },
      {
        name: 'Redis Cache',
        status: 'healthy',
        url: 'redis://localhost:6379',
        version: '7.0',
        uptime: '2h 15m',
        lastCheck: new Date().toISOString(),
      },
      {
        name: 'Prometheus',
        status: 'healthy',
        url: 'http://localhost:9090',
        version: '2.48.1',
        uptime: '2h 15m',
        lastCheck: new Date().toISOString(),
      },
      {
        name: 'Grafana',
        status: 'healthy',
        url: 'http://localhost:3000',
        version: '10.2.3',
        uptime: '2h 15m',
        lastCheck: new Date().toISOString(),
      },
      {
        name: 'Jaeger',
        status: 'healthy',
        url: 'http://localhost:16686',
        version: '1.52',
        uptime: '2h 15m',
        lastCheck: new Date().toISOString(),
      },
      {
        name: 'ML Model Service',
        status: 'degraded',
        url: 'internal',
        version: '1.0.0',
        uptime: '1h 45m',
        lastCheck: new Date().toISOString(),
        issue: 'High prediction latency detected',
      },
    ]);

    // Mock system metrics
    setSystemMetrics({
      cpu: {
        usage: 35.2,
        cores: 8,
        load_avg: [1.2, 1.1, 1.0],
      },
      memory: {
        total: 16000000000, // 16GB
        used: 768000000,   // 768MB
        available: 15232000000,
        cached: 2048000000,
      },
      disk: {
        total: 500000000000, // 500GB
        used: 229000000000,  // 229GB
        available: 271000000000,
        io_read: 1024000,    // KB/s
        io_write: 512000,    // KB/s
      },
      network: {
        rx_bytes: 1048576000, // 1GB
        tx_bytes: 524288000,  // 512MB
        rx_packets: 1000000,
        tx_packets: 800000,
      },
    });
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy': return <CheckCircle />;
      case 'degraded': return <Warning />;
      case 'unhealthy': return <Error />;
      default: return <Error />;
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatUptime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  const getUsageColor = (percentage) => {
    if (percentage < 50) return 'success';
    if (percentage < 80) return 'warning';
    return 'error';
  };

  const overallHealth = healthData.status;
  const healthySystems = Object.values(healthData.components || {}).filter(s => s === 'healthy').length;
  const totalSystems = Object.keys(healthData.components || {}).length;

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        🏥 System Health
      </Typography>

      {/* Overall Health Status */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h5">Overall System Health</Typography>
            <Chip
              icon={getStatusIcon(overallHealth)}
              label={`${overallHealth?.toUpperCase()} (${healthySystems}/${totalSystems} systems)`}
              color={getStatusColor(overallHealth)}
              size="large"
            />
          </Box>
          
          {overallHealth === 'degraded' && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              Some systems are experiencing issues. Check individual components below.
            </Alert>
          )}
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Computer sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                <Typography variant="h6">API Server</Typography>
                <Typography variant="body2" color="text.secondary">
                  Uptime: {formatUptime(healthData.metrics?.uptime || 0)}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Speed sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
                <Typography variant="h6">Performance</Typography>
                <Typography variant="body2" color="text.secondary">
                  Avg Response: {healthData.metrics?.response_time || 0}ms
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Security sx={{ fontSize: 40, color: 'info.main', mb: 1 }} />
                <Typography variant="h6">Security</Typography>
                <Typography variant="body2" color="text.secondary">
                  All systems secured
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Cloud sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
                <Typography variant="h6">Observability</Typography>
                <Typography variant="body2" color="text.secondary">
                  Monitoring active
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* System Resources */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Computer sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h6">CPU Usage</Typography>
              </Box>
              <Typography variant="h4" color={getUsageColor(systemMetrics.cpu?.usage || 0)}>
                {(systemMetrics.cpu?.usage || 0).toFixed(1)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={systemMetrics.cpu?.usage || 0}
                color={getUsageColor(systemMetrics.cpu?.usage || 0)}
                sx={{ mt: 1, mb: 1 }}
              />
              <Typography variant="body2" color="text.secondary">
                {systemMetrics.cpu?.cores || 0} cores available
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Memory sx={{ mr: 1, color: 'info.main' }} />
                <Typography variant="h6">Memory</Typography>
              </Box>
              <Typography variant="h4" color={getUsageColor((systemMetrics.memory?.used / systemMetrics.memory?.total) * 100 || 0)}>
                {formatBytes(systemMetrics.memory?.used || 0)}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={(systemMetrics.memory?.used / systemMetrics.memory?.total) * 100 || 0}
                color={getUsageColor((systemMetrics.memory?.used / systemMetrics.memory?.total) * 100 || 0)}
                sx={{ mt: 1, mb: 1 }}
              />
              <Typography variant="body2" color="text.secondary">
                of {formatBytes(systemMetrics.memory?.total || 0)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Storage sx={{ mr: 1, color: 'warning.main' }} />
                <Typography variant="h6">Disk Usage</Typography>
              </Box>
              <Typography variant="h4" color={getUsageColor((systemMetrics.disk?.used / systemMetrics.disk?.total) * 100 || 0)}>
                {formatBytes(systemMetrics.disk?.used || 0)}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={(systemMetrics.disk?.used / systemMetrics.disk?.total) * 100 || 0}
                color={getUsageColor((systemMetrics.disk?.used / systemMetrics.disk?.total) * 100 || 0)}
                sx={{ mt: 1, mb: 1 }}
              />
              <Typography variant="body2" color="text.secondary">
                of {formatBytes(systemMetrics.disk?.total || 0)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Speed sx={{ mr: 1, color: 'success.main' }} />
                <Typography variant="h6">Network I/O</Typography>
              </Box>
              <Typography variant="body1">
                ↓ {formatBytes(systemMetrics.network?.rx_bytes || 0)}
              </Typography>
              <Typography variant="body1">
                ↑ {formatBytes(systemMetrics.network?.tx_bytes || 0)}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Total transferred
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Component Status */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Component Health Status
          </Typography>
          
          <Grid container spacing={2}>
            {Object.entries(healthData.components || {}).map(([component, status]) => (
              <Grid item xs={12} sm={6} md={4} key={component}>
                <Paper sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Typography variant="body1" sx={{ textTransform: 'capitalize' }}>
                    {component.replace('_', ' ')}
                  </Typography>
                  <Chip
                    icon={getStatusIcon(status)}
                    label={status}
                    color={getStatusColor(status)}
                    size="small"
                  />
                </Paper>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Services Status */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Service Status
          </Typography>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Service</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Version</TableCell>
                  <TableCell>Uptime</TableCell>
                  <TableCell>URL</TableCell>
                  <TableCell>Last Check</TableCell>
                  <TableCell>Issues</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {services.map((service) => (
                  <TableRow key={service.name}>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                        {service.name}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={getStatusIcon(service.status)}
                        label={service.status}
                        color={getStatusColor(service.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {service.version}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {service.uptime}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                        {service.url}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {new Date(service.lastCheck).toLocaleTimeString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {service.issue && (
                        <Chip
                          label={service.issue}
                          color="warning"
                          size="small"
                          variant="outlined"
                        />
                      )}
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

export default SystemHealth;