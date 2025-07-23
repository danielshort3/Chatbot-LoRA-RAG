FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime


WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

# ----- dependency layer -----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----- source code -----
COPY . .
RUN pip install --no-cache-dir .


EXPOSE 7860
CMD ["python", "-m", "vgj_chat"]
