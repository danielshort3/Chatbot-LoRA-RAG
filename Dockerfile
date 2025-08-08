################  Base image without TorchServe  ###############
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

################  System deps ##################################
# Install git and clean up apt cache in the same layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama CLI for dataset generation
RUN curl -L https://ollama.com/download/ollama-linux-amd64 -o /tmp/ollama.tar.gz && \
    tar -xzf /tmp/ollama.tar.gz -C /usr/local/bin && \
    rm /tmp/ollama.tar.gz

################  Install Python deps  #########################
COPY requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt && rm /tmp/req.txt

################  Copy application code  ########################
WORKDIR /app
COPY pyproject.toml .
COPY README.md .
COPY vgj_chat ./vgj_chat
RUN pip install --no-cache-dir .

################  Expose port & launch  ########################
EXPOSE 8080
