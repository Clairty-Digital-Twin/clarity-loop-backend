import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
  Divider,
} from '@mui/material';
import {
  Dashboard,
  ShowChart,
  Timeline,
  Description,
  Warning,
  HealthAndSafety,
  Visibility,
} from '@mui/icons-material';

const drawerWidth = 280;

const menuItems = [
  {
    text: 'Dashboard',
    icon: <Dashboard />,
    path: '/dashboard',
    description: 'Overview and key metrics',
  },
  {
    text: 'Metrics',
    icon: <ShowChart />,
    path: '/metrics',
    description: 'Performance metrics',
  },
  {
    text: 'Traces',
    icon: <Timeline />,
    path: '/traces',
    description: 'Distributed tracing',
  },
  {
    text: 'Logs',
    icon: <Description />,
    path: '/logs',
    description: 'Application logs',
  },
  {
    text: 'Alerts',
    icon: <Warning />,
    path: '/alerts',
    description: 'Alert management',
  },
  {
    text: 'System Health',
    icon: <HealthAndSafety />,
    path: '/health',
    description: 'System status',
  },
];

function Sidebar({ darkMode }) {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          background: darkMode 
            ? 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)'
            : 'linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%)',
          borderRight: darkMode ? '1px solid #333' : '1px solid #e0e0e0',
        },
      }}
    >
      {/* Logo and Title */}
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
          <Visibility sx={{ fontSize: 32, mr: 1, color: 'primary.main' }} />
          <Typography variant="h5" component="div" sx={{ fontWeight: 'bold' }}>
            Clarity
          </Typography>
        </Box>
        <Typography variant="body2" color="textSecondary">
          Observability Dashboard
        </Typography>
      </Box>

      <Divider />

      {/* Navigation Menu */}
      <List sx={{ px: 2, py: 1 }}>
        {menuItems.map((item) => {
          const isSelected = location.pathname === item.path;
          
          return (
            <ListItem key={item.text} disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton
                onClick={() => navigate(item.path)}
                selected={isSelected}
                sx={{
                  borderRadius: 2,
                  mb: 0.5,
                  '&.Mui-selected': {
                    backgroundColor: darkMode 
                      ? 'rgba(144, 202, 249, 0.12)'
                      : 'rgba(25, 118, 210, 0.12)',
                    '&:hover': {
                      backgroundColor: darkMode 
                        ? 'rgba(144, 202, 249, 0.2)'
                        : 'rgba(25, 118, 210, 0.2)',
                    },
                  },
                  '&:hover': {
                    backgroundColor: darkMode 
                      ? 'rgba(255, 255, 255, 0.08)'
                      : 'rgba(0, 0, 0, 0.04)',
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    color: isSelected ? 'primary.main' : 'text.secondary',
                    minWidth: 40,
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Typography 
                      variant="body1" 
                      sx={{ 
                        fontWeight: isSelected ? 600 : 400,
                        color: isSelected ? 'primary.main' : 'text.primary',
                      }}
                    >
                      {item.text}
                    </Typography>
                  }
                  secondary={
                    <Typography variant="caption" color="textSecondary">
                      {item.description}
                    </Typography>
                  }
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>

      <Box sx={{ flexGrow: 1 }} />

      {/* Footer */}
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Divider sx={{ mb: 2 }} />
        <Typography variant="caption" color="textSecondary" display="block">
          Clarity Digital Twin Platform
        </Typography>
        <Typography variant="caption" color="textSecondary" display="block">
          v0.2.0
        </Typography>
        <Typography variant="caption" color="textSecondary" display="block" sx={{ mt: 1 }}>
          🤖 Built with AI
        </Typography>
      </Box>
    </Drawer>
  );
}

export default Sidebar;