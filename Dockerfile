FROM python:3.10-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements_api.txt .

RUN pip install --no-cache-dir -r requirements_api.txt

COPY . .
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "src.api"]