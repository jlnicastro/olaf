FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 libsm6 libxrender1 libxext6 libsqlite3-dev git \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install -r requirements.txt --system

COPY . .

EXPOSE 8501

# New entrypoint runs KB setup then launches app
CMD ["bash", "-c", "streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
