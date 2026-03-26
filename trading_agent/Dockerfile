FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common/ ./common/
COPY trading_agent/src/ ./src/
COPY data_fetcher/ ./data_fetcher/
COPY models/ ./models/
CMD ["python", "src/agent.py"]
