# Railway Deployment Commands for Prod Branch

## Quick Deploy Commands

### 1. Commit Changes
```bash
git add .
git commit -m "Optimize Railway deployment - reduce image size and fix auth timeout"
```

### 2. Push to Prod Branch
```bash
git push origin prod
```

### 3. Alternative: Force Deploy via Railway CLI (if needed)
```bash
# Install Railway CLI if not installed
npm install -g @railway/cli

# Login to Railway
railway login

# Link to your project
railway link

# Deploy from current branch
railway up --detach
```

## Key Optimizations Made

1. **Created `nixpacks.toml`** - Railway's recommended deployment method
2. **Optimized `requirements.txt`** - Pinned versions, CPU-only PyTorch, removed unnecessary deps
3. **Created `.dockerignore`** - Reduces build context size
4. **Updated `.railwayignore`** - Excludes all checkpoint files (~300MB saved)

## Expected Results
- **Build time**: Reduced from 10+ minutes to 3-5 minutes
- **Image size**: Reduced from 7.8GB to ~200-500MB
- **Deploy success**: Should pass authentication step

## If Still Failing
Try these Railway dashboard settings:
1. Go to Settings → Environment
2. Set: `NIXPACKS_PYTHON_VERSION=3.11`
3. Set: `RAILWAY_DOCKERFILE_PATH=` (leave empty to use nixpacks)
4. Redeploy

## Production Features Included
- ✅ 3 ML Models (MFCC, Mel CNN, Raw CNN)
- ✅ VAD (Voice Activity Detection)
- ✅ Flask web interface
- ✅ Gunicorn production server
- ❌ Training checkpoints (excluded)
- ❌ Heavy transformers (wav2vec2, whisper excluded)
