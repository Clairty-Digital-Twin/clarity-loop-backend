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
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Grid,
  Paper,
} from '@mui/material';
import {
  CheckCircle,
  Close,
  OpenInNew,
  Warning,
  Error,
  Info,
} from '@mui/icons-material';

function Alerts({ socket, realTimeData }) {
  const [alerts, setAlerts] = useState([]);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [stats, setStats] = useState({});

  useEffect(() => {
    // Mock alert data
    const mockAlerts = [
      {
        id: 'alert_001',
        rule_name: 'high_error_rate',
        severity: 'critical',
        status: 'firing',
        message: 'API error rate is above 1% threshold',
        fired_at: new Date(Date.now() - 300000).toISOString(), // 5 minutes ago
        labels: {
          service: 'clarity-backend',
          endpoint: '/api/v1/health-data',
        },
        annotations: {
          runbook_url: 'https://docs.clarity.com/runbooks/high-error-rate',
          description: 'The API is returning errors at a rate higher than acceptable threshold',
        },
      },
      {
        id: 'alert_002',
        rule_name: 'slow_response_time',
        severity: 'warning',
        status: 'resolved',
        message: '99th percentile response time above 1 second',
        fired_at: new Date(Date.now() - 1800000).toISOString(), // 30 minutes ago
        resolved_at: new Date(Date.now() - 600000).toISOString(), // 10 minutes ago
        labels: {
          service: 'clarity-backend',
          method: 'POST',
        },
        annotations: {
          runbook_url: 'https://docs.clarity.com/runbooks/slow-response-time',
        },
      },
      {
        id: 'alert_003',
        rule_name: 'ml_model_error_rate',
        severity: 'warning',
        status: 'firing',
        message: 'ML model error rate is above 5%',
        fired_at: new Date(Date.now() - 900000).toISOString(), // 15 minutes ago
        labels: {
          service: 'clarity-backend',
          model: 'PAT-transformer',
        },
        annotations: {
          runbook_url: 'https://docs.clarity.com/runbooks/ml-model-errors',
        },
      },
    ];
    
    setAlerts(mockAlerts);
    
    // Mock stats
    setStats({
      total_alerts: 15,
      active_alerts: 2,
      critical_alerts: 1,
      warning_alerts: 1,
      resolved_today: 8,
    });
  }, []);

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'error';
      case 'warning': return 'warning';
      case 'info': return 'info';
      default: return 'default';
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'critical': return <Error />;
      case 'warning': return <Warning />;
      case 'info': return <Info />;
      default: return <Info />;
    }
  };

  const getStatusColor = (status) => {
    return status === 'resolved' ? 'success' : 'error';
  };

  const handleResolveAlert = async (alertId) => {
    try {
      // Mock API call
      console.log('Resolving alert:', alertId);
      
      // Update local state
      setAlerts(prev => prev.map(alert => 
        alert.id === alertId 
          ? { ...alert, status: 'resolved', resolved_at: new Date().toISOString() }
          : alert
      ));
    } catch (error) {
      console.error('Failed to resolve alert:', error);
    }
  };

  const handleViewDetails = (alert) => {
    setSelectedAlert(alert);
    setDialogOpen(true);
  };

  const activeAlerts = alerts.filter(alert => alert.status === 'firing');
  const resolvedAlerts = alerts.filter(alert => alert.status === 'resolved');

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        🚨 Alert Management
      </Typography>

      {/* Alert Statistics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2.4}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" color="primary">{stats.total_alerts}</Typography>
            <Typography variant="body2">Total Alerts</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={2.4}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" color="error">{stats.active_alerts}</Typography>
            <Typography variant="body2">Active</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={2.4}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" color="error">{stats.critical_alerts}</Typography>
            <Typography variant="body2">Critical</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={2.4}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" color="warning">{stats.warning_alerts}</Typography>
            <Typography variant="body2">Warning</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={2.4}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" color="success">{stats.resolved_today}</Typography>
            <Typography variant="body2">Resolved Today</Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Active Alerts */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom color="error">
            🔥 Active Alerts ({activeAlerts.length})
          </Typography>
          
          {activeAlerts.length === 0 ? (
            <Alert severity="success">
              <Typography>No active alerts. All systems are operating normally.</Typography>
            </Alert>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Alert</TableCell>
                    <TableCell>Severity</TableCell>
                    <TableCell>Message</TableCell>
                    <TableCell>Fired At</TableCell>
                    <TableCell>Duration</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {activeAlerts.map((alert) => (
                    <TableRow key={alert.id}>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                          {alert.rule_name}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          icon={getSeverityIcon(alert.severity)}
                          label={alert.severity}
                          color={getSeverityColor(alert.severity)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {alert.message}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {new Date(alert.fired_at).toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {Math.floor((Date.now() - new Date(alert.fired_at).getTime()) / 60000)} min
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Button
                            size="small"
                            variant="contained"
                            color="success"
                            startIcon={<CheckCircle />}
                            onClick={() => handleResolveAlert(alert.id)}
                          >
                            Resolve
                          </Button>
                          <IconButton
                            size="small"
                            onClick={() => handleViewDetails(alert)}
                          >
                            <OpenInNew />
                          </IconButton>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Recent Resolved Alerts */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom color="success">
            ✅ Recently Resolved ({resolvedAlerts.length})
          </Typography>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Alert</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Message</TableCell>
                  <TableCell>Fired At</TableCell>
                  <TableCell>Resolved At</TableCell>
                  <TableCell>Duration</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {resolvedAlerts.map((alert) => (
                  <TableRow key={alert.id} sx={{ opacity: 0.7 }}>
                    <TableCell>
                      <Typography variant="body2">
                        {alert.rule_name}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={getSeverityIcon(alert.severity)}
                        label={alert.severity}
                        color={getSeverityColor(alert.severity)}
                        size="small"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {alert.message}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {new Date(alert.fired_at).toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {alert.resolved_at ? new Date(alert.resolved_at).toLocaleString() : 'N/A'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {alert.resolved_at 
                          ? Math.floor((new Date(alert.resolved_at).getTime() - new Date(alert.fired_at).getTime()) / 60000) + ' min'
                          : 'N/A'
                        }
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <IconButton
                        size="small"
                        onClick={() => handleViewDetails(alert)}
                      >
                        <OpenInNew />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Alert Details Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
        {selectedAlert && (
          <>
            <DialogTitle>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6">{selectedAlert.rule_name}</Typography>
                <IconButton onClick={() => setDialogOpen(false)}>
                  <Close />
                </IconButton>
              </Box>
            </DialogTitle>
            <DialogContent>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>Status</Typography>
                  <Chip 
                    label={selectedAlert.status} 
                    color={getStatusColor(selectedAlert.status)}
                    sx={{ mb: 2 }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>Severity</Typography>
                  <Chip 
                    icon={getSeverityIcon(selectedAlert.severity)}
                    label={selectedAlert.severity} 
                    color={getSeverityColor(selectedAlert.severity)}
                    sx={{ mb: 2 }}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>Message</Typography>
                  <Typography variant="body1" sx={{ mb: 2 }}>
                    {selectedAlert.message}
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>Labels</Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                    {Object.entries(selectedAlert.labels).map(([key, value]) => (
                      <Chip
                        key={key}
                        label={`${key}: ${value}`}
                        variant="outlined"
                        size="small"
                      />
                    ))}
                  </Box>
                </Grid>
                {selectedAlert.annotations.runbook_url && (
                  <Grid item xs={12}>
                    <Button
                      variant="outlined"
                      startIcon={<OpenInNew />}
                      onClick={() => window.open(selectedAlert.annotations.runbook_url, '_blank')}
                    >
                      View Runbook
                    </Button>
                  </Grid>
                )}
              </Grid>
            </DialogContent>
            <DialogActions>
              {selectedAlert.status === 'firing' && (
                <Button
                  variant="contained"
                  color="success"
                  startIcon={<CheckCircle />}
                  onClick={() => {
                    handleResolveAlert(selectedAlert.id);
                    setDialogOpen(false);
                  }}
                >
                  Resolve Alert
                </Button>
              )}
              <Button onClick={() => setDialogOpen(false)}>
                Close
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
}

export default Alerts;