# Base Image: Official Python 3.12 Slim (Debian Bookworm)
FROM python:3.12-slim-bookworm

# Set environment variables to prevent pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Add /app to pythonpath so imports work natively
ENV PYTHONPATH=/app
# Ensure the compiled C-library is visible to Python
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH"

WORKDIR /app

# --- SYSTEM DEPENDENCIES & TA-LIB COMPILATION ---
# We install build tools, compile TA-Lib, then remove the source to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- PYTHON DEPENDENCIES ---
COPY Pipfile Pipfile.lock ./

# Install pipenv and dependencies (System-wide, no venv)
# Note: We use --system to install into the container's global python
RUN pip install --no-cache-dir pipenv && \
    pipenv install --system --deploy

# --- APPLICATION CODE ---
COPY src/ ./src/
COPY main.py .

# --- RUNTIME ---
CMD ["python", "main.py"]
