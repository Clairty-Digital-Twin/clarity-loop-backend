import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Chip,
  Tooltip,
  Badge,
} from '@mui/material';
import {
  DarkMode,
  LightMode,
  Refresh,
  Settings,
  Circle,
  Notifications,
} from '@mui/icons-material';

const drawerWidth = 280;

function Header({ darkMode, toggleDarkMode, isConnected, realTimeData }) {
  const connectionStatus = isConnected ? 'Connected' : 'Disconnected';
  const connectionColor = isConnected ? 'success' : 'error';
  const activeAlerts = realTimeData.active_alerts || 0;

  return (
    <AppBar
      position="fixed"
      sx={{
        width: `calc(100% - ${drawerWidth}px)`,
        ml: `${drawerWidth}px`,
        backgroundColor: darkMode ? 'background.paper' : 'background.paper',
        color: 'text.primary',
        boxShadow: darkMode
          ? '0 2px 4px rgba(0, 0, 0, 0.3)'
          : '0 2px 4px rgba(0, 0, 0, 0.1)',
        borderBottom: `1px solid ${darkMode ? '#333' : '#e0e0e0'}`,
      }}
    >
      <Toolbar>
        {/* Connection Status */}
        <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
          <Circle
            sx={{
              fontSize: 12,
              color: isConnected ? 'success.main' : 'error.main',
              mr: 1,
            }}
          />
          <Chip
            label={connectionStatus}
            size="small"
            color={connectionColor}
            variant={isConnected ? 'filled' : 'outlined'}
          />
        </Box>

        {/* Real-time Data Indicators */}
        {realTimeData.lastUpdate && (
          <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
            <Typography variant="body2" color="textSecondary" sx={{ mr: 1 }}>
              Last update:
            </Typography>
            <Typography variant="body2" color="primary">
              {new Date(realTimeData.lastUpdate * 1000).toLocaleTimeString()}
            </Typography>
          </Box>
        )}

        <Box sx={{ flexGrow: 1 }} />

        {/* Alerts Indicator */}
        <Tooltip title={`${activeAlerts} active alerts`}>
          <IconButton color="inherit" sx={{ mr: 1 }}>
            <Badge badgeContent={activeAlerts} color="error" max={99}>
              <Notifications />
            </Badge>
          </IconButton>
        </Tooltip>

        {/* Refresh Button */}
        <Tooltip title="Refresh data">
          <IconButton
            color="inherit"
            onClick={() => window.location.reload()}
            sx={{ mr: 1 }}
          >
            <Refresh />
          </IconButton>
        </Tooltip>

        {/* Settings Button */}
        <Tooltip title="Settings">
          <IconButton color="inherit" sx={{ mr: 1 }}>
            <Settings />
          </IconButton>
        </Tooltip>

        {/* Dark Mode Toggle */}
        <Tooltip title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}>
          <IconButton color="inherit" onClick={toggleDarkMode}>
            {darkMode ? <LightMode /> : <DarkMode />}
          </IconButton>
        </Tooltip>
      </Toolbar>
    </AppBar>
  );
}

export default Header;