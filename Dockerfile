# Use a Python image that includes PyTorch support (without Jetson-specific CUDA dependencies)
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt first to leverage Docker cache (faster rebuilds)
COPY requirements.txt /app/

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the application code (e.g., Streamlit app, etc.) into the container
COPY . /app

# Expose the port for Streamlit or your application (port 7860 for Streamlit)
EXPOSE 7860

# Run your app
CMD ["streamlit", "run", "app.py", "--server.enableCORS=false", "--server.port=7860"]
