-- PostgreSQL Development Database Initialization
-- Creates necessary extensions and tables for local development

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create development schemas
CREATE SCHEMA IF NOT EXISTS clarity_dev;
CREATE SCHEMA IF NOT EXISTS clarity_analytics;

-- Set search path
SET search_path TO clarity_dev, public;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    date_of_birth DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    preferences JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE
);

-- Create health_data table
CREATE TABLE IF NOT EXISTS health_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) REFERENCES users(user_id),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    heart_rate INTEGER,
    steps INTEGER,
    sleep_hours DECIMAL(4,2),
    sleep_quality VARCHAR(20),
    activity_type VARCHAR(50),
    data_source VARCHAR(100),
    device_type VARCHAR(50),
    blood_pressure_systolic INTEGER,
    blood_pressure_diastolic INTEGER,
    weight_kg DECIMAL(5,2),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_health_data_user_timestamp ON health_data(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_health_data_timestamp ON health_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_health_data_activity_type ON health_data(activity_type);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- Create insights table
CREATE TABLE IF NOT EXISTS insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) REFERENCES users(user_id),
    insight_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    confidence_score DECIMAL(4,3),
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source VARCHAR(100),
    metadata JSONB DEFAULT '{}'
);

-- Create analytics views
CREATE OR REPLACE VIEW clarity_analytics.daily_user_stats AS
SELECT 
    user_id,
    DATE(timestamp) as date,
    AVG(heart_rate) as avg_heart_rate,
    MAX(steps) as max_steps,
    AVG(sleep_hours) as avg_sleep_hours,
    COUNT(*) as data_points
FROM health_data 
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY user_id, DATE(timestamp);

-- Create development functions
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data
INSERT INTO users (user_id, email, first_name, last_name, preferences) 
VALUES 
    ('admin@clarity.dev', 'admin@clarity.dev', 'Admin', 'User', '{"role": "admin", "notifications": true}'),
    ('testuser@clarity.dev', 'testuser@clarity.dev', 'Test', 'User', '{"notifications": true, "units": "metric"}')
ON CONFLICT (user_id) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA clarity_dev TO clarity_dev;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA clarity_dev TO clarity_dev;
GRANT USAGE ON SCHEMA clarity_analytics TO clarity_dev;
GRANT SELECT ON ALL TABLES IN SCHEMA clarity_analytics TO clarity_dev;

-- Development utilities
CREATE OR REPLACE FUNCTION reset_dev_data() RETURNS void AS $$
BEGIN
    DELETE FROM insights;
    DELETE FROM health_data;
    DELETE FROM users WHERE user_id NOT IN ('admin@clarity.dev', 'testuser@clarity.dev');
    RAISE NOTICE 'Development data reset complete';
END;
$$ LANGUAGE plpgsql;