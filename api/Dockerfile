FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common/ ./common/
COPY api/src/ ./src/
COPY data_fetcher/ ./data_fetcher/
COPY models/ ./models/
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
