FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

# Install dependencies including espeak
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak

# Copy requirements and install Python packages
COPY requirements.txt .
RUN uv pip install -r requirements.txt --system

# Copy your app code
COPY . .

EXPOSE 8501

CMD ["bash", "-c", "python knowledge_base.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
