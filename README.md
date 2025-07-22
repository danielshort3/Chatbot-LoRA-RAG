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

## Dependencies

The application requires `bitsandbytes` and `faiss-gpu` in addition to the
standard dependencies listed in `pyproject.toml`.

## Docker

A Dockerfile is provided for a fully containerised setup. Building the image
will run the crawler and indexer so the demo is ready to launch:

```bash
docker build -t vgj-chat .
docker run -p 7860:7860 vgj-chat
```

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
