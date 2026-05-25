FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY Code/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY Code/app.py .
COPY Code/config.py .
COPY Code/train_model_v2.py .
COPY Code/resnet18_bloodgroup.pth .
COPY Code/templates/ templates/
COPY Code/static/ static/

# Create upload directory
RUN mkdir -p static/uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV PORT=10000

# Expose port (Render uses 10000)
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/')" || exit 1

# Run with gunicorn (Render sets PORT env var)
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app
