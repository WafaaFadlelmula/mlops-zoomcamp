FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY dummy_metrics_calculation.py .

CMD ["python", "dummy_metrics_calculation.py"]