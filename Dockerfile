FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# ----- dependency layer -----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----- source code -----
COPY . .
RUN pip install --no-cache-dir .

# Optionally run your data/index creation here,
# or leave it to entrypoint/startup.

EXPOSE 7860
CMD ["python", "-m", "vgj_chat"]
