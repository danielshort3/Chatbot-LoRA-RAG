# Chatbot-LoRa-RAG

Example RAG chatbot using a LoRA adapted language model.  The code is
organised as a Python package named `vgj_chat`.

Launch the Gradio demo with:

```bash
python -m vgj_chat
```

## Configuration

Configuration defaults live in `vgj_chat.config.Config`.  Any field can be
overridden by environment variables prefixed with `VGJ_` or by passing a
command-line option of the same name.

Environment variables use upper-case field names, for example
`VGJ_INDEX_PATH` overrides `index_path`.

Command-line overrides replace underscores with dashes, e.g.:

```bash
python -m vgj_chat --index-path my.index --top-k 3
```

Both methods may be combined; CLI options take precedence.
