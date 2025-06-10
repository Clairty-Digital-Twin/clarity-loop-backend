# 🚀 CLARITY Backend Setup Guide for Beginners

Welcome! This guide will help you get the CLARITY backend running on AWS.

## 📋 Prerequisites

1. **Python 3.11+** installed
2. **Docker** installed
3. **AWS Account** (free tier is fine)
4. **Gemini API Key** (free from Google)

## 🔧 Quick Start (Local Development)

### Step 1: Clone and Setup

```bash
# Navigate to your project
cd clarity-loop-backend

# Install Python dependencies
pip install -e .

# Copy environment template
cp .env.example .env
```

### Step 2: Configure Your API Key

1. Get your Gemini API key:
   - Go to https://aistudio.google.com/apikey
   - Click "Create API Key"
   - Copy the key

2. Add to your `.env` file:
   ```bash
   # Open .env in your editor
   # Find this line:
   GEMINI_API_KEY=your-gemini-api-key-here
   
   # Replace with your actual key:
   GEMINI_API_KEY=AIzaSyB-actualkey123...
   ```

### Step 3: Run with Mock Services (No AWS needed!)

```bash
# Make sure this is set in your .env:
SKIP_EXTERNAL_SERVICES=true

# Run the server
python -m clarity.main_aws
```

Your server is now running at http://localhost:8080! 🎉

## 🌩️ AWS Deployment (When You're Ready)

### Step 1: Build Docker Image

```bash
# Build the AWS-optimized image
docker build -f Dockerfile.aws.clean -t clarity-backend:latest .

# Test it locally
docker run -p 8000:8000 --env-file .env clarity-backend:latest
```

### Step 2: Deploy to AWS

```bash
# Run the deployment script
./deploy-to-aws.sh
```

## 📁 Where Things Go

```
clarity-loop-backend/
├── .env                    # Your API keys (NEVER commit this!)
├── .env.example           # Template for .env
├── secrets/               # JSON credential files only
│   └── README.md         # More security info
├── src/                  # Application code
└── Dockerfile.aws.clean  # AWS deployment container
```

## 🔐 Security Best Practices

### For API Keys:
- **Development**: Store in `.env` file
- **Production**: Use AWS Secrets Manager

### Golden Rules:
1. ✅ Check `.env` is in `.gitignore`
2. ❌ Never commit `.env` to git
3. ❌ Never share API keys in code
4. ✅ Use environment variables

## 🆘 Troubleshooting

### "Module not found" error
```bash
pip install -e .
```

### "GEMINI_API_KEY not set" error
- Check your `.env` file has the key
- Make sure you copied `.env.example` to `.env`

### AWS Connection Issues
- Set `SKIP_EXTERNAL_SERVICES=true` for local development
- You don't need AWS services to test the Gemini AI features!

## 📚 Learning Path

1. **Start Local**: Use mock services to learn the API
2. **Add Gemini**: Test AI features with your API key
3. **Try Docker**: Build and run containers locally
4. **Deploy to AWS**: When you're ready for production

## 🎯 Next Steps

1. Test the health endpoint:
   ```bash
   curl http://localhost:8080/health
   ```

2. Check the API docs:
   - http://localhost:8080/docs

3. Try the Gemini insights endpoint:
   ```bash
   curl -X POST http://localhost:8080/api/v1/insights/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Analyze my sleep patterns"}'
   ```

---

Remember: You're doing great! Take it step by step. 🌟