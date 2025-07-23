# Chatbot-LoRa-RAG

Example RAG chatbot using a LoRA adapted language model.  The code is
organised as a Python package named `vgj_chat`.

Launch the Gradio demo with:

```bash
python -m vgj_chat --hf-token <HF_TOKEN>
```

## Quick start

Create a local environment with [Hatch](https://hatch.pypa.io/) and run the demo.
The default base model `mistralai/Mistral-7B-Instruct-v0.2` is gated on
[Hugging Face](https://huggingface.co/). Request access on the model page and
generate an access token from your account settings. Pass this token using the
`--hf-token` command-line option or the `VGJ_HF_TOKEN` environment variable.

Run the demo with:

```bash
pipx run hatch env create
pipx run hatch run python -m vgj_chat --hf-token <HF_TOKEN>
```
The environment installs the GPU-enabled FAISS package so the demo can
use the GPU when available.  The Docker image installs the
CUDA-enabled `bitsandbytes` and `faiss-gpu-cu12` wheels.  If a matching wheel
isn't available for your Python or CUDA version you will need the
`cuda-toolkit` headers to compile them from source (e.g.
`apt install cuda-toolkit-12-8`).

## Dependencies

The application requires `bitsandbytes` and the GPU-enabled `faiss-gpu-cu12`
package in addition to the standard dependencies listed in `pyproject.toml`.

FAISS uses the GPU by default when available. Set `VGJ_FAISS_CUDA=false` or
pass `--faiss-cuda false` to force CPU indexing while keeping the rest of the
application on the GPU.

## LoRA adapter

This repository does not include the fine‑tuned LoRA adapter weights. Point the
application at your own checkpoint by passing `--lora-dir` or setting the
`VGJ_LORA_DIR` environment variable:

```bash
python -m vgj_chat --lora-dir path/to/lora-checkpoint
# or
VGJ_LORA_DIR=path/to/lora-checkpoint python -m vgj_chat
```

Run `scripts/finetune.py` to train a new adapter when suitable data is
available.

## Docker

A Dockerfile is provided for a fully containerised setup. The image
uses the PyTorch 2.7.1 base with CUDA 12.8 and cuDNN 9. Building it will
run the crawler and indexer so the demo is ready to launch:

```bash
docker build -t vgj-chat .
# GPU acceleration requires the host to install the NVIDIA Container Toolkit
# and have compatible NVIDIA drivers. Run the container with:
docker run --gpus all -p 7860:7860 -e VGJ_HF_TOKEN=<token> vgj-chat
```

### GPU compatibility issues

If the container exits with an error similar to:

```
Faiss assertion 'err__ == cudaSuccess' ... CUDA error 209 no kernel image is available for execution on the device
```

the FAISS wheel was built for a GPU architecture that does not match your
hardware. Disable FAISS GPU usage while keeping the model on the GPU with:

```bash
docker run -p 7860:7860 -e VGJ_HF_TOKEN=<token> -e VGJ_FAISS_CUDA=false vgj-chat
```

If you need FAISS acceleration, rebuild the Docker image with FAISS compiled for
your GPU.

## Architecture

```
vgj_chat
├── cli.py            # CLI entrypoint launching the UI
├── config.py         # dataclass with configuration defaults
├── ui/gradio_app.py  # builds the Gradio interface
├── data/             # crawling, indexing and dataset helpers
├── models/           # RAG model, LoRA fine-tuning and guardrails
└── __main__.py       # allows `python -m vgj_chat`
```

## Configuration

Configuration defaults live in `vgj_chat.config.Config`.  Any field can be
overridden by environment variables prefixed with `VGJ_` or by passing a
command-line option of the same name.

Environment variables use upper-case field names, for example
`VGJ_INDEX_PATH` overrides `index_path`.

Command-line overrides replace underscores with dashes, e.g.:

```bash
python -m vgj_chat --hf-token <HF_TOKEN> --index-path my.index --top-k 3
```

Debug logging is disabled by default. Enable it with `--debug true` or set
`VGJ_DEBUG=true` in the environment.

Both methods may be combined; CLI options take precedence.

## Development

Run the helpers in the `Makefile` to format, lint and test the project:

```bash
make format  # run black and isort
make lint    # run ruff
make test    # run pytest
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
