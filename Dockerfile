FROM python:3.11-slim

WORKDIR /app

# Install dependencies (no heavy system deps needed for ONNX)
COPY Code/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY Code/app_deploy.py ./app.py
COPY Code/config.py .
COPY Code/resnet18_bloodgroup.onnx .
COPY Code/templates/ templates/
COPY Code/static/ static/

# Create upload directory
RUN mkdir -p static/uploads

# Environment
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV PORT=10000

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/')" || exit 1

# Single worker to save memory on free tier
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
