FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY setup.py .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install .

EXPOSE 8000

CMD ["python", "-m", "fraud_model.predict"]