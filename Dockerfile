FROM pytorch/pytorch:2.2.2-cuda12.4-cudnn8-devel

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        faiss-gpu \
        bitsandbytes \
        accelerate \
        git+https://github.com/huggingface/transformers.git && \
    pip install --no-cache-dir .

RUN python scripts/crawl.py && \
    python scripts/build_index.py

EXPOSE 7860
CMD ["python", "-m", "vgj_chat"]
