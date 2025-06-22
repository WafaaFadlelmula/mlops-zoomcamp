FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Copy your score.py script
COPY score.py .

# Install dependencies inside the container
RUN pip install --no-cache-dir scikit-learn==1.5.0 pandas pyarrow


ENTRYPOINT ["python", "score.py"]
