# Use a slim Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app_local

# Install system dependencies (for building the app)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly (including xgboost)
RUN pip install --no-cache-dir \
    dash \
    dash-bootstrap-components \
    plotly \
    pandas \
    numpy \
    joblib \
    xgboost \
    scikit-learn

# Copy the app and model files into the container
COPY app_local.py /app_local/
COPY models /app_local/models
COPY model_inference_example.py /app_local/

# Expose port 8050 for the Dash app
EXPOSE 8050

# Set the default command to run the app
CMD ["python", "app_local.py"]
