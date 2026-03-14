# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for compiling some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# Disable cache to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports: 8000 for FastAPI, 8501 for Streamlit
EXPOSE 8000
EXPOSE 8501

# Entrypoint script to run both services or pass a command
# For simplicity, we create a small bash script inline to run both if needed,
# or we just rely on docker-compose for multi-container deployments.
# Let's provide a script to run FastAPI by default.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
