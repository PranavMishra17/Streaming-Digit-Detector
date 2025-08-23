# Use Python 3.9 as recommended by HF Spaces
FROM python:3.9-slim

# Create user for HF Spaces (required)
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (as user, limited packages)
# Note: HF Spaces has restrictions on system packages
COPY --chown=user ./requirements_hf.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application files (essential files only)
COPY --chown=user ./app.py ./app.py
COPY --chown=user ./audio_processors ./audio_processors
COPY --chown=user ./utils ./utils
COPY --chown=user ./models ./models

# Copy environment template (users can set their own HF_TOKEN)
COPY --chown=user ./.env.example ./.env

# Create log directory
RUN mkdir -p /app/logs

# Expose port (HF Spaces requires 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/api/health').raise_for_status()" || exit 1

# Run the application
CMD ["python", "app.py"]