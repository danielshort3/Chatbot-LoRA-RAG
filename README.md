# Chatbot-LoRa-RAG

A lightweight Retrieval-Augmented Generation (RAG) demo that fine‑tunes a language model with LoRA, merges the adapter into a 4‑bit gpt‑oss‑20B model and exposes a FastAPI endpoint for inference. The code lives in the `vgj_chat` package and includes helper scripts for crawling pages, building an index and training or merging the adapter.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
# install a CUDA build of PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev]

# required for 4-bit quantisation
pip install bitsandbytes
```

### Docker (recommended)

Skip manual setup by running the project in Docker:

```bash
docker build -t vgj-chat .
docker run --gpus all -p 8080:8080 -e VGJ_HF_TOKEN=<token> vgj-chat
# run on the CPU instead of the GPU
docker run -p 8080:8080 -e VGJ_CUDA=false -e VGJ_HF_TOKEN=<token> vgj-chat
```

## Quick start

Start the FastAPI server to perform local inference:

```bash
python serve.py
```

Send a request:

```bash
curl -X POST localhost:8080/invocations -H 'Content-Type: application/json' \
    -d '{"inputs": "What is there to do in Grand Junction?"}'
```

## Pipeline

Run all preparation steps (crawl pages, build the index, fine‑tune and merge) in one command:

```bash
python scripts/run_pipeline.py
```

Set `VGJ_HF_TOKEN` so the scripts can download the base model. Each script in `scripts/` can also be run individually.

Alternatively execute the individual steps in `scripts/` if you need more control.

Scripts overview:

1. `crawl.py` – download webpages
2. `build_index.py` – create the FAISS index
3. `build_dataset.py` – generate training pairs
4. `finetune.py` – train the LoRA adapter
5. `merge_lora.py` – merge the adapter into a 4‑bit model

Run the standalone trainer on a JSONL Q&A file to produce a LoRA checkpoint:

```bash
python scripts/finetune.py --data data/qa.jsonl --output-dir data/lora-vgj-checkpoint
```

Pass `--config config.yaml` to load hyper‑parameters from a YAML file.

The pipeline builds the FAISS index on the CPU so no special GPU support is
required. All other stages—including auto‑generated Q&A creation, LoRA
fine‑tuning and inference—will use CUDA when available.

### Ollama model for dataset generation

`build_dataset.py` queries the `gpt-oss:20b` model via [Ollama](https://ollama.com/).
Pull the model on the host and mount your local models directory when running in
Docker so it can be reused instead of downloaded again:

```bash
ollama pull gpt-oss:20b  # on the host machine

# run the pipeline with the models dir mounted
docker run --gpus all \
  -e HF_TOKEN \
  -v "$HOME/.ollama:/root/.ollama" \
  -v "$(pwd)":/workspace vgj-chat \
  /bin/bash -c "cd /workspace && python scripts/run_pipeline.py"
```

On Windows hosts the models may live under `C:\Users\<user>\.ollama\models`
or `%LOCALAPPDATA%\Ollama\models`. When running from WSL this path usually
looks like `/mnt/c/Users/<user>/.ollama/models`, which should be mounted to
`/root/.ollama` in the container:

```bash
docker run --gpus all \
  -e HF_TOKEN \
  -v /mnt/c/Users/<user>/.ollama/models:/root/.ollama \
  -v "$(pwd)":/workspace vgj-chat \
  /bin/bash -c "cd /workspace && python scripts/run_pipeline.py"
```

Set `OLLAMA_DEBUG=1` to print diagnostic information (such as the resolved
models directory and the list of available models) if the dataset builder cannot
locate `gpt-oss:20b`.

## Using a LoRA adapter

After fine‑tuning the adapter it can be merged into a 4‑bit quantized model using `scripts/merge_lora.py`. The resulting model directory is loaded automatically when the FastAPI server starts. Set `VGJ_LORA_DIR` if you want to load a different checkpoint before merging and `VGJ_MERGED_MODEL_DIR` to change the directory used for inference.

## Docker

Use `docker exec` to run the helper scripts inside the container as needed. The quick start section shows how to build and run the image.

## SageMaker deployment

After running the pipeline and producing `faiss.index`, `meta.jsonl` and the
merged 4‑bit model directory, build `Dockerfile.sagemaker` and push the image to
ECR. Copy the artifacts into a `model/` directory before building so the
container includes everything required for inference on SageMaker. Dependency
installation uses `requirements.sagemaker.txt` during the build.

## Configuration

Default settings live in `vgj_chat.config.Config`. Override any option via environment variables prefixed `VGJ_`. Example:

```bash
VGJ_INDEX_PATH=my.index VGJ_TOP_K=3 VGJ_DEBUG=true python serve.py
```

## Contributing

Format, lint and test the project with the helpers in the `Makefile`:

```bash
make format  # run black and isort
make lint    # run ruff
make test    # run pytest
```

Pull requests are welcome. CI runs the same checks on every push.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
