FROM python:3.9-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpoppler-cpp-dev \
    pkg-config \
    tesseract-ocr \
    libreoffice \
    pandoc \
    ghostscript \
    poppler-utils \
    antiword \
    tesseract-ocr-all \
    git \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 8000


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]