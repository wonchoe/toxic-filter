FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && pip install --upgrade pip \
 && pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y --auto-remove gcc \
 && rm -rf /root/.cache /var/lib/apt/lists/* /tmp/*

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]