# Chatbot-LoRa-RAG

A small Retrieval-Augmented Generation (RAG) demo that fine-tunes a language
model with LoRA and serves a Gradio chat UI. The application lives in the
`vgj_chat` Python package and includes helper scripts to build the dataset and
index used for retrieval.

Launch the demo with:

```bash
python -m vgj_chat --hf-token <HF_TOKEN>
```

## Quick start

1. Install [Hatch](https://hatch.pypa.io/) via `pipx install hatch`.
2. Create a local environment and run the UI:

```bash
pipx run hatch env create
pipx run hatch run python -m vgj_chat --hf-token <HF_TOKEN>
```

The default model `mistralai/Mistral-7B-Instruct-v0.2` is gated on
[Hugging Face](https://huggingface.co/). Request access on the model page and
supply your token with `--hf-token` or the `VGJ_HF_TOKEN` environment variable.

## Dependencies

Apart from the packages listed in `pyproject.toml`, the demo requires
`bitsandbytes` and the GPU-enabled `faiss-gpu-cu12` wheel. The optional
fine-tuning script depends on the `trl` library.

FAISS uses the GPU by default when available. Set `VGJ_FAISS_CUDA=false` or pass
`--faiss-cuda false` to force CPU indexing while keeping the model on the GPU.

## Repository structure

```
vgj_chat/           Python package with the main application
scripts/            Helper scripts for crawling, indexing and training
archive/            Notebook with early experiments
tests/              Pytest suite
.github/workflows/  Continuous integration pipeline
```

Key modules inside `vgj_chat`:

```
cli.py            CLI entrypoint launching the UI
config.py         Configuration dataclass
ui/gradio_app.py  Builds the Gradio interface
data/             Crawling, indexing and dataset helpers
models/           RAG model, LoRA fine-tuning and guardrails
__main__.py       Enables `python -m vgj_chat`
```

## Preparation

Run the helper scripts in order or execute the pipeline in one go:

```bash
python scripts/run_pipeline.py --limit 20
```

Manual steps if you prefer running them individually:

1. `python scripts/crawl.py` – download webpages
2. `python scripts/build_index.py` – create the FAISS index
3. `python scripts/build_dataset.py` – build training pairs
4. `python scripts/finetune.py` – train a LoRA adapter
5. `python -m vgj_chat` – start chatting

Each script accepts `--help` for available options. Export `VGJ_HF_TOKEN` so the
scripts can download the base model. A CUDA‑enabled GPU is recommended for
indexing and fine‑tuning but the steps will fall back to the CPU if necessary.

## LoRA adapter

The repository does not include fine‑tuned adapter weights. Point the chat
application at your checkpoint with:

```bash
python -m vgj_chat --lora-dir path/to/lora
```

or set `VGJ_LORA_DIR` in the environment. Run `scripts/finetune.py` whenever you
have new training data.

## Compare mode

Start the UI with `--compare` to show answers from the LoRA+FAISS pipeline and a
raw baseline side by side:

```bash
python -m vgj_chat --hf-token <TOKEN> --compare
```

## Docker

A Dockerfile is provided for a containerised setup based on the PyTorch 2.7.1
CUDA 12.8 image. Build and run it as follows:

```bash
docker build -t vgj-chat .
# GPU acceleration requires the NVIDIA Container Toolkit and compatible drivers
docker run --gpus all -p 7860:7860 -e VGJ_HF_TOKEN=<token> vgj-chat
```

Execute the helper scripts inside the running container to crawl pages, create
an index and fine‑tune the adapter:

```bash
docker exec -it <container> python scripts/run_pipeline.py --limit 20
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

## Configuration

Configuration defaults live in `vgj_chat.config.Config`. Override any field with
an environment variable prefixed `VGJ_` or with a command-line option of the same
name. CLI options use dashes instead of underscores:

```bash
python -m vgj_chat --hf-token <HF_TOKEN> --index-path my.index --top-k 3
```

Debug logging is disabled by default. Enable it with `--debug true` or
`VGJ_DEBUG=true`.

## Development

Format, lint and test the project using the helpers in `Makefile`:

```bash
make format  # run black and isort
make lint    # run ruff
make test    # run pytest
```

A GitHub Actions workflow runs the same checks on every push and pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
more information.
