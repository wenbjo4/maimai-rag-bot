FROM python:3.10-slim

WORKDIR /app

COPY app/ .
COPY data/ ./data/

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
