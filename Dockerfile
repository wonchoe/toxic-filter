FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

# Встановлюємо gcc, pip, а потім видаляємо gcc
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && pip install --upgrade pip \
 && pip install --no-cache-dir torch==2.1.2+cpu -f https://download.pytorch.org/whl/torch_stable.html \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y --auto-remove gcc \
 && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
