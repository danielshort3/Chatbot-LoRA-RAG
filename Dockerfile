################  Base image without TorchServe  ###############
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

################  Install Python deps  #########################
COPY requirements.sagemaker.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt && rm /tmp/req.txt

################  Copy application code  ########################
WORKDIR /app
COPY serve.py .
COPY inference.py .
COPY gradio_vgj_chat.py .
COPY pyproject.toml .
COPY README.md .
COPY vgj_chat ./vgj_chat
RUN pip install --no-cache-dir .

################  Expose port & launch  ########################
EXPOSE 8080
CMD ["python", "serve.py"]
