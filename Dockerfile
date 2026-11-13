FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /service
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data ./data
COPY evaluation ./evaluation
COPY load ./load
RUN mkdir -p data/corpus/source data/corpus/processed data/index

EXPOSE 8000 8101 8102
