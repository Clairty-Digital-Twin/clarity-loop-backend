import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import io from 'socket.io-client';

import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import Metrics from './pages/Metrics';
import Traces from './pages/Traces';
import Logs from './pages/Logs';
import Alerts from './pages/Alerts';
import SystemHealth from './pages/SystemHealth';

// Create theme with dark mode support
const createAppTheme = (mode) => createTheme({
  palette: {
    mode,
    primary: {
      main: mode === 'dark' ? '#90caf9' : '#1976d2',
    },
    secondary: {
      main: mode === 'dark' ? '#f48fb1' : '#dc004e',
    },
    background: {
      default: mode === 'dark' ? '#121212' : '#f5f5f5',
      paper: mode === 'dark' ? '#1e1e1e' : '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: mode === 'dark' 
            ? '0 4px 6px rgba(0, 0, 0, 0.3)' 
            : '0 2px 4px rgba(0, 0, 0, 0.1)',
        },
      },
    },
  },
});

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : true; // Default to dark mode
  });
  
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [realTimeData, setRealTimeData] = useState({});

  // Toggle dark mode
  const toggleDarkMode = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    localStorage.setItem('darkMode', JSON.stringify(newMode));
  };

  // Setup WebSocket connection for real-time updates
  useEffect(() => {
    const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
    const newSocket = io(`${apiBaseUrl}/api/v1/observability/stream`);
    
    newSocket.on('connect', () => {
      console.log('Connected to observability stream');
      setIsConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from observability stream');
      setIsConnected(false);
    });

    newSocket.on('connected', (data) => {
      console.log('Observability stream connected:', data.message);
    });

    newSocket.on('metrics_update', (data) => {
      setRealTimeData(prev => ({
        ...prev,
        ...data.data,
        lastUpdate: data.timestamp
      }));
    });

    newSocket.on('error', (error) => {
      console.error('WebSocket error:', error);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  const theme = createAppTheme(darkMode ? 'dark' : 'light');

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
          <Sidebar darkMode={darkMode} />
          <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
            <Header 
              darkMode={darkMode} 
              toggleDarkMode={toggleDarkMode}
              isConnected={isConnected}
              realTimeData={realTimeData}
            />
            <Box component="main" sx={{ flexGrow: 1, p: 3, mt: 8 }}>
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={
                  <Dashboard 
                    socket={socket} 
                    realTimeData={realTimeData} 
                    isConnected={isConnected}
                  />
                } />
                <Route path="/metrics" element={
                  <Metrics 
                    socket={socket} 
                    realTimeData={realTimeData}
                  />
                } />
                <Route path="/traces" element={
                  <Traces 
                    socket={socket} 
                    realTimeData={realTimeData}
                  />
                } />
                <Route path="/logs" element={
                  <Logs 
                    socket={socket} 
                    realTimeData={realTimeData}
                  />
                } />
                <Route path="/alerts" element={
                  <Alerts 
                    socket={socket} 
                    realTimeData={realTimeData}
                  />
                } />
                <Route path="/health" element={
                  <SystemHealth 
                    socket={socket} 
                    realTimeData={realTimeData}
                  />
                } />
              </Routes>
            </Box>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;