# Deployment Guide: Streaming Digit Classifier

## Overview

This guide explains how to deploy the Streaming Digit Classifier using a split architecture:

- **Frontend**: Vercel (Static HTML/CSS/JS)
- **Backend API**: Hugging Face Spaces (Flask + Docker)

## Architecture

```
Frontend (Vercel)     -->     Backend API (HF Spaces)
- HTML/CSS/JS                 - Flask REST API  
- Audio recording             - ML Models (PyTorch)
- Real-time UI               - Audio processing
- Retro styling              - Docker container
```

## Prerequisites

1. **Accounts needed**:
   - [Vercel](https://vercel.com) account
   - [Hugging Face](https://huggingface.co) account
   - GitHub account (for automated deployments)

2. **Local requirements**:
   - Git
   - Node.js 18+ (for Vercel CLI)
   - Python 3.9+ (for testing)

## Backend Deployment (Hugging Face Spaces)

### Step 1: Create HF Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **Space name**: `streaming-digit-classifier` (or your preferred name)
   - **SDK**: Docker
   - **Hardware**: CPU Basic (free tier)
   - **Visibility**: Public

### Step 2: Push Backend Code

```bash
# Clone your new HF Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/streaming-digit-classifier
cd streaming-digit-classifier

# Copy backend files from this repo
cp /path/to/this/repo/app.py .
cp /path/to/this/repo/Dockerfile .
cp /path/to/this/repo/requirements_hf.txt .
cp /path/to/this/repo/.env.example .

# Copy essential directories
cp -r /path/to/this/repo/audio_processors ./
cp -r /path/to/this/repo/utils ./
cp -r /path/to/this/repo/models ./  # Only copy best models

# Create README.md for your HF Space
cat > README.md << 'EOF'
---
title: Streaming Digit Classifier API
emoji: 🎤
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Streaming Digit Classifier API

Backend API for real-time spoken digit recognition (0-9).

## Features

- ML Models: MFCC + Dense NN, Mel CNN, Raw CNN
- External API integration (Whisper)
- Real-time audio processing
- RESTful API endpoints

## API Endpoints

- `GET /` - API status
- `POST /api/process_audio` - Process audio file
- `POST /api/process_audio_chunk` - Process streaming chunk
- `GET /api/health` - Health check
- `GET /api/processors` - Available processors

Frontend: [Deployed on Vercel](https://your-frontend-url.vercel.app)
EOF

# Commit and push
git add .
git commit -m "Initial backend deployment"
git push
```

### Step 3: Configure Environment

1. In your HF Space settings, add environment variables if needed:
   - `HF_TOKEN` - Your Hugging Face token (for external APIs)
   - `SECRET_KEY` - Flask secret key

### Step 4: Monitor Deployment

1. Your Space will build automatically (takes 5-15 minutes)
2. Check the logs in the HF Space interface
3. Once deployed, note your Space URL: `https://YOUR_USERNAME-streaming-digit-classifier.hf.space`

## Frontend Deployment (Vercel)

### Step 1: Prepare Frontend

```bash
# From this repo's hf branch
cd frontend

# Update API URL in index.html
# Replace 'https://paranoiid-streaming-digit-classifier.hf.space' 
# with your actual HF Space URL
```

### Step 2: Deploy to Vercel

#### Option A: Vercel CLI (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy from frontend directory
cd frontend
vercel

# Follow prompts:
# - Link to existing project or create new
# - Choose your Vercel team/account
# - Confirm deployment settings

# For production deployment
vercel --prod
```

#### Option B: GitHub Integration

1. Push frontend to GitHub repository
2. Go to [Vercel Dashboard](https://vercel.com/dashboard)
3. Click "Import Project" 
4. Select your GitHub repo
5. Set:
   - **Root Directory**: `frontend`
   - **Build Command**: (leave empty for static site)
   - **Output Directory**: (leave default)

### Step 3: Configure CORS

Update your backend's CORS settings to allow your Vercel domain:

```python
# In app.py, update CORS origins
CORS(app, origins=['https://your-frontend-url.vercel.app'])
```

Redeploy your HF Space after this change.

## GitHub Actions Deployment (Optional)

Automated deployment is configured in `.github/workflows/deploy.yml`.

### Setup Secrets

In your GitHub repository settings, add these secrets:

1. **Vercel secrets**:
   - `VERCEL_TOKEN` - Get from [Vercel Settings](https://vercel.com/account/tokens)
   - `VERCEL_ORG_ID` - From your Vercel team settings
   - `VERCEL_PROJECT_ID` - From your Vercel project settings

2. **HF secrets**:
   - `HF_TOKEN` - Your Hugging Face access token

### Trigger Deployment

```bash
# Push to hf branch triggers automatic frontend deployment
git push origin hf

# Manual deployment (includes backend instructions)
# Go to Actions tab in GitHub and run "Deploy to Vercel and HF Spaces"
```

## Testing the Deployment

### Backend API Test

```bash
# Test health endpoint
curl https://YOUR_USERNAME-streaming-digit-classifier.hf.space/api/health

# Test processors
curl https://YOUR_USERNAME-streaming-digit-classifier.hf.space/api/processors

# Test audio processing (requires audio file)
curl -X POST \
  -F "audio=@test.wav" \
  -F "method=ml_mfcc" \
  https://YOUR_USERNAME-streaming-digit-classifier.hf.space/api/process_audio
```

### Frontend Test

1. Open your Vercel URL in browser
2. Check API connection status in the UI
3. Grant microphone permissions
4. Test recording and digit recognition

## Troubleshooting

### Backend Issues

1. **Build failures**:
   - Check HF Space logs
   - Verify `requirements_hf.txt` dependencies
   - Ensure model files are present

2. **Runtime errors**:
   - Check model file paths
   - Verify audio processing dependencies
   - Monitor memory usage (HF Spaces has limits)

3. **CORS errors**:
   - Update `CORS(app, origins=[...])` with your Vercel URL
   - Ensure preflight requests are handled

### Frontend Issues

1. **API connection failed**:
   - Verify HF Space URL in `index.html`
   - Check CORS configuration
   - Test API endpoints manually

2. **Microphone access**:
   - Ensure HTTPS (required for microphone)
   - Check browser permissions
   - Test on different browsers/devices

3. **Build failures**:
   - Check Vercel build logs
   - Verify file structure in `frontend/` directory

## Performance Optimization

### Backend (HF Spaces)

1. **Model optimization**:
   - Use CPU-optimized PyTorch models
   - Quantize models if needed
   - Cache model loading

2. **Memory management**:
   - Monitor memory usage
   - Implement request timeouts
   - Use efficient audio processing

### Frontend (Vercel)

1. **Caching**:
   - Static assets cached automatically
   - Configure cache headers in `vercel.json`

2. **Loading optimization**:
   - Minimize JavaScript bundle size
   - Lazy load components if needed

## Monitoring

### HF Spaces
- Check Space logs for errors
- Monitor resource usage
- Set up uptime monitoring

### Vercel
- Use Vercel Analytics
- Monitor function execution (if using)
- Check deployment logs

## Costs

- **HF Spaces**: Free CPU Basic tier available
- **Vercel**: Free tier includes 100GB bandwidth/month
- **Total**: $0/month for basic usage

## Scaling

### High Traffic
- Upgrade HF Space to GPU/larger CPU
- Use Vercel Pro for better performance
- Consider CDN for static assets

### Multiple Regions
- Deploy HF Spaces in different regions
- Use Vercel Edge Network (automatic)

## Security

1. **API Keys**: Store in environment variables
2. **CORS**: Restrict origins in production  
3. **Rate Limiting**: Implement on backend if needed
4. **HTTPS**: Enabled by default on both platforms

## Maintenance

1. **Dependencies**: Keep updated for security
2. **Models**: Retrain periodically for accuracy
3. **Monitoring**: Set up alerts for downtime
4. **Backups**: HF Spaces handles git-based backup

## Support

- **HF Spaces**: [Documentation](https://huggingface.co/docs/hub/spaces)
- **Vercel**: [Documentation](https://vercel.com/docs)
- **This Project**: Create GitHub issues

---

Happy deploying! 🚀