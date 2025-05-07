FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y gcc \
 && pip install --upgrade pip \
 && pip install --no-cache-dir 'numpy<2' \
 && pip install --no-cache-dir torch==2.1.2+cpu -f https://download.pytorch.org/whl/torch_stable.html \
 && pip install --no-cache-dir -r requirements.txt


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]