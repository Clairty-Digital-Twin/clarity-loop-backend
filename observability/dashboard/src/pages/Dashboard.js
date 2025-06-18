import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  CircularProgress,
  Alert,
  Paper,
} from '@mui/material';
import {
  TrendingUp,
  Error,
  Speed,
  People,
  Warning,
  CheckCircle,
  Memory,
  Cpu,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

// Utility function to format numbers
const formatNumber = (num) => {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
  return num.toFixed(0);
};

// Utility function to format percentages
const formatPercent = (num) => `${(num * 100).toFixed(2)}%`;

// Metric card component
const MetricCard = ({ title, value, unit, icon: Icon, color = 'primary', trend, loading = false }) => (
  <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
    <CardContent sx={{ flexGrow: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Typography color="textSecondary" gutterBottom variant="h6">
          {title}
        </Typography>
        <Icon color={color} />
      </Box>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
          <CircularProgress size={24} />
        </Box>
      ) : (
        <>
          <Typography variant="h4" component="div" sx={{ fontWeight: 'bold', mb: 1 }}>
            {typeof value === 'number' ? formatNumber(value) : value}
            {unit && <Typography component="span" variant="h6" color="textSecondary"> {unit}</Typography>}
          </Typography>
          
          {trend && (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <TrendingUp 
                color={trend > 0 ? 'success' : 'error'} 
                sx={{ fontSize: 16, mr: 0.5 }} 
              />
              <Typography 
                variant="body2" 
                color={trend > 0 ? 'success.main' : 'error.main'}
              >
                {trend > 0 ? '+' : ''}{trend.toFixed(1)}%
              </Typography>
            </Box>
          )}
        </>
      )}
    </CardContent>
  </Card>
);

// Status indicator component
const StatusIndicator = ({ status, label }) => {
  const getColor = () => {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'error';
      default: return 'default';
    }
  };

  const getIcon = () => {
    switch (status) {
      case 'healthy': return <CheckCircle />;
      case 'degraded': return <Warning />;
      case 'unhealthy': return <Error />;
      default: return null;
    }
  };

  return (
    <Chip
      icon={getIcon()}
      label={label || status}
      color={getColor()}
      variant="outlined"
      sx={{ textTransform: 'capitalize' }}
    />
  );
};

function Dashboard({ socket, realTimeData, isConnected }) {
  const [summary, setSummary] = useState({});
  const [health, setHealth] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [metricsHistory, setMetricsHistory] = useState([]);

  // Fetch dashboard data
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Fetch summary data
        const [summaryRes, healthRes, alertsRes] = await Promise.all([
          fetch('/api/v1/observability/metrics/summary'),
          fetch('/api/v1/observability/health'),
          fetch('/api/v1/observability/alerts?limit=10'),
        ]);

        if (!summaryRes.ok || !healthRes.ok || !alertsRes.ok) {
          throw new Error('Failed to fetch dashboard data');
        }

        const [summaryData, healthData, alertsData] = await Promise.all([
          summaryRes.json(),
          healthRes.json(),
          alertsRes.json(),
        ]);

        setSummary(summaryData);
        setHealth(healthData);
        setAlerts(alertsData);
        setError(null);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
    
    // Refresh data every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  // Update metrics history with real-time data
  useEffect(() => {
    if (realTimeData.lastUpdate) {
      setMetricsHistory(prev => {
        const newEntry = {
          timestamp: realTimeData.lastUpdate,
          active_alerts: realTimeData.active_alerts || 0,
          response_time: Math.random() * 0.5 + 0.1, // Mock data
          error_rate: Math.random() * 0.02, // Mock data
          requests_per_second: Math.random() * 50 + 20, // Mock data
        };
        
        // Keep only last 50 data points
        const updated = [...prev, newEntry].slice(-50);
        return updated;
      });
    }
  }, [realTimeData]);

  // Sample data for charts
  const responseTimeData = metricsHistory.map(item => ({
    timestamp: new Date(item.timestamp * 1000).toLocaleTimeString(),
    value: item.response_time * 1000, // Convert to ms
  }));

  const requestRateData = metricsHistory.map(item => ({
    timestamp: new Date(item.timestamp * 1000).toLocaleTimeString(),
    value: item.requests_per_second,
  }));

  const statusDistribution = [
    { name: '2xx', value: 85, fill: '#4caf50' },
    { name: '4xx', value: 12, fill: '#ff9800' },
    { name: '5xx', value: 3, fill: '#f44336' },
  ];

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress size={48} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Error loading dashboard: {error}
      </Alert>
    );
  }

  return (
    <Box>
      {/* Connection Status */}
      <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Typography variant="h4" component="h1" sx={{ flexGrow: 1 }}>
          🔍 Clarity Observability Dashboard
        </Typography>
        <StatusIndicator 
          status={isConnected ? 'healthy' : 'unhealthy'} 
          label={isConnected ? 'Connected' : 'Disconnected'} 
        />
        {realTimeData.lastUpdate && (
          <Typography variant="body2" color="textSecondary">
            Last update: {new Date(realTimeData.lastUpdate * 1000).toLocaleTimeString()}
          </Typography>
        )}
      </Box>

      {/* Key Metrics Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Requests"
            value={summary.total_requests || 0}
            icon={TrendingUp}
            color="primary"
            trend={5.2}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Error Rate"
            value={summary.error_rate ? formatPercent(summary.error_rate / 100) : '0%'}
            icon={Error}
            color={summary.error_rate > 1 ? 'error' : 'success'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Avg Response Time"
            value={summary.avg_response_time || 0}
            unit="ms"
            icon={Speed}
            color="info"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Alerts"
            value={summary.active_alerts || realTimeData.active_alerts || 0}
            icon={Warning}
            color={summary.active_alerts > 0 ? 'warning' : 'success'}
          />
        </Grid>
      </Grid>

      {/* System Health and Charts Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* System Health */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: 300 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Health
              </Typography>
              <Box sx={{ mt: 2 }}>
                <StatusIndicator status={health.status} label={`Overall: ${health.status}`} />
                <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {health.components && Object.entries(health.components).map(([component, status]) => (
                    <Box key={component} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                        {component}
                      </Typography>
                      <StatusIndicator status={status} />
                    </Box>
                  ))}
                </Box>
                
                {health.metrics && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Key Metrics
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Cpu fontSize="small" />
                        <Typography variant="body2">
                          CPU: {(health.metrics.cpu_usage || 0).toFixed(1)}%
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Memory fontSize="small" />
                        <Typography variant="body2">
                          Memory: {formatNumber(health.metrics.memory_usage || 0)}B
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <People fontSize="small" />
                        <Typography variant="body2">
                          Connections: {health.metrics.active_connections || 0}
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Response Time Chart */}
        <Grid item xs={12} md={8}>
          <Card sx={{ height: 300 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Response Time (ms)
              </Typography>
              <Box sx={{ height: 220 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={responseTimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Area type="monotone" dataKey="value" stroke="#1976d2" fill="#1976d2" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Request Rate and Status Distribution Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Request Rate Chart */}
        <Grid item xs={12} md={8}>
          <Card sx={{ height: 300 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Request Rate (req/s)
              </Typography>
              <Box sx={{ height: 220 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={requestRateData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="value" stroke="#4caf50" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Status Distribution */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: 300 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Status Code Distribution
              </Typography>
              <Box sx={{ height: 220, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={statusDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}%`}
                    >
                      {statusDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Alerts */}
      {alerts.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Recent Alerts
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {alerts.slice(0, 5).map((alert, index) => (
                <Paper 
                  key={alert.id || index} 
                  sx={{ 
                    p: 2, 
                    border: '1px solid',
                    borderColor: alert.severity === 'critical' ? 'error.main' : 
                                alert.severity === 'warning' ? 'warning.main' : 'info.main',
                    borderRadius: 2,
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                      {alert.rule_name}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <StatusIndicator status={alert.severity} />
                      <StatusIndicator status={alert.status} />
                    </Box>
                  </Box>
                  <Typography variant="body2" color="textSecondary">
                    {alert.message}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    {new Date(alert.fired_at).toLocaleString()}
                  </Typography>
                </Paper>
              ))}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default Dashboard;