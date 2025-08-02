# Chatbot-LoRa-RAG

A lightweight Retrieval-Augmented Generation (RAG) demo that fine‑tunes a language model with LoRA, merges the adapter into a 4‑bit Mistral‑7B model and serves a Gradio chat UI. The code lives in the `vgj_chat` package and includes helper scripts for crawling pages, building an index and training or merging the adapter.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# required for 4-bit quantisation
pip install bitsandbytes
```

### Docker (recommended)

Skip manual setup by running the project in Docker:

```bash
docker build -t vgj-chat .
docker run --gpus all -p 7860:7860 -e VGJ_HF_TOKEN=<token> vgj-chat
```

## Quick start

Launch the chat UI with your Hugging Face token. The generator loads a merged
4‑bit model by default:

```bash
python -m vgj_chat --hf-token <HF_TOKEN>
```

The default model `mistralai/Mistral-7B-Instruct-v0.2` is gated on Hugging Face, so the token is mandatory.

## Pipeline

Run all preparation steps (crawl pages, build the index, fine‑tune and merge) in one command:

```bash
VGJ_FAISS_CUDA=false python scripts/run_pipeline.py
```

Set `VGJ_HF_TOKEN` so the scripts can download the base model. Each script in `scripts/` can also be run individually.

Alternatively execute the individual steps in `scripts/` if you need more control.

Scripts overview:

1. `crawl.py` – download webpages
2. `build_index.py` – create the FAISS index
3. `build_dataset.py` – generate training pairs
4. `finetune.py` – train the LoRA adapter
5. `merge_lora.py` – merge the adapter into a 4‑bit model

FAISS GPU acceleration has not been tested due to incompatible hardware. Set `VGJ_FAISS_CUDA=false` (or `--faiss-cuda false`) to run indexing on the CPU while the rest of the pipeline uses CUDA.

## Using a LoRA adapter

After fine‑tuning the adapter it can be merged into a 4‑bit quantized model using `scripts/merge_lora.py`. The resulting model directory is loaded automatically when you launch the chat UI. Set `VGJ_LORA_DIR` if you want to load a different checkpoint before merging and `VGJ_MERGED_MODEL_DIR` to change the directory used for inference.

## Compare mode

Add `--compare` to start a dual-chat UI showing answers from the LoRA+FAISS pipeline beside a raw baseline:

```bash
python -m vgj_chat --hf-token <HF_TOKEN> --compare
```

## Docker

Use `docker exec` to run the helper scripts inside the container as needed. The quick start section shows how to build and run the image.

## SageMaker deployment

After running the pipeline and producing `faiss.index`, `meta.jsonl` and the
merged 4‑bit model directory, build `Dockerfile.sagemaker` and push the image to
ECR. Copy the artifacts into a `model/` directory before building so the
container includes everything required for inference on SageMaker.

## Configuration

Default settings live in `vgj_chat.config.Config`. Override any option via environment variables prefixed `VGJ_` or with the matching CLI flag. Example:

```bash
python -m vgj_chat --index-path my.index --top-k 3 --debug true
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
