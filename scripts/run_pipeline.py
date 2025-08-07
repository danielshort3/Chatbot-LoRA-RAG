import argparse
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import torch  # type: ignore

from vgj_chat.models.rag.retrieval import SentenceWindowRetriever
from vgj_chat.models.rag import boot as _boot
from vgj_chat.utils.text import token_len

CRAWL_TXT_DIR = Path("data/html_txt")
AUTO_QA_JL = Path("data/dataset/vgj_auto_dataset.jsonl")
LORA_DIR = Path("data/lora-vgj-checkpoint")
INDEX_PATH = Path("data/faiss.index")
META_PATH = Path("data/meta.jsonl")

# ──────────────────────────────────────────────────────────
# NEW CONSTANTS
MERGED_SRC = Path("data/mistral-merged-4bit")
DEST_DIR = Path("model")  # or Path("models") if you prefer
ARCHIVE = Path("model.tar.gz")  # will live in project root
# ──────────────────────────────────────────────────────────

CTX_TOK_LIMIT = 1500
OUT_TOK_LIMIT = 350


def _answer(question: str) -> str:
    """Generate an answer for *question* using retrieved context blocks."""

    retriever = SentenceWindowRetriever()
    raw_blocks = retriever.retrieve_windows(question)

    context_blocks: list[str] = []
    seen: set[tuple[str, str]] = set()
    for block in raw_blocks:
        header, _body = block.split("\n", 1)
        doc_match = re.search(r"<DOC_ID:(\d+)>", header)
        para_match = re.search(r"<PARA_ID:(\d+)>", header)
        key = (
            doc_match.group(1) if doc_match else "",
            para_match.group(1) if para_match else "",
        )
        if key in seen:
            continue
        candidate = "\n\n".join(context_blocks + [block])
        if token_len(candidate) > CTX_TOK_LIMIT:
            break
        context_blocks.append(block)
        seen.add(key)

    if not context_blocks:
        return "Sorry, I couldn’t find anything relevant."

    context = "\n\n".join(context_blocks)

    _boot._ensure_boot()
    assert _boot.CHAT

    tok = _boot.CHAT.tokenizer
    model = _boot.CHAT.model

    prompt = f"{context}\n\nQ: {question}\nA:"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    generated = model.generate(
        **inputs,
        max_new_tokens=min(OUT_TOK_LIMIT, _boot.CFG.max_new_tokens),
        do_sample=False,
    )

    return tok.decode(generated[0], skip_special_tokens=True)[len(prompt):].strip()


def main() -> None:
    # ---------- existing CLI and pipeline code ----------
    parser = argparse.ArgumentParser(description="Run full pipeline")

    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of pages to crawl"
    )
    parser.add_argument(
        "--question", type=str, default=None, help="Optional question to answer"
    )
    args = parser.parse_args()

    steps: list[list[str]] = []
    if args.limit is None and any(CRAWL_TXT_DIR.glob("*.txt")):
        print(f"{CRAWL_TXT_DIR} populated; skipping crawl step")
    else:
        crawl_cmd = ["python", "scripts/crawl.py"]
        if args.limit is not None:
            crawl_cmd.extend(["--limit", str(args.limit)])
        steps.append(crawl_cmd)

    if INDEX_PATH.exists() and META_PATH.exists():
        print("FAISS index and metadata already exist; skipping build")
    else:
        steps.append(["python", "scripts/build_index.py"])

    if AUTO_QA_JL.exists():
        print(f"{AUTO_QA_JL} exists; skipping dataset build")
    else:
        steps.append(["python", "scripts/build_dataset.py"])

    if (LORA_DIR / "adapter_model.safetensors").exists():
        print(f"{LORA_DIR} exists; skipping fine-tune step")
    else:
        steps.append(["python", "scripts/finetune.py", "--data", str(AUTO_QA_JL)])

    if (MERGED_SRC / "model.safetensors").exists():
        print(f"{MERGED_SRC} exists; skipping merge step")
    else:
        steps.append(["python", "scripts/merge_lora.py"])

    for cmd in steps:
        subprocess.run(cmd, check=True)

    # ---------- NEW FILE-GATHERING LOGIC ----------
    if ARCHIVE.exists():
        print(f"{ARCHIVE} exists; skipping archive step")
    else:
        # 1. Prep clean DEST_DIR
        if DEST_DIR.exists():
            shutil.rmtree(DEST_DIR)
        DEST_DIR.mkdir(parents=True, exist_ok=True)

        # 2. Copy merged model files into DEST_DIR
        if MERGED_SRC.exists():
            for src_path in MERGED_SRC.iterdir():
                shutil.copy2(src_path, DEST_DIR / src_path.name)
        else:
            raise FileNotFoundError(
                f"{MERGED_SRC} not found – make sure merge_lora.py completed."
            )

        # 3. Copy index & metadata
        shutil.copy2(INDEX_PATH, DEST_DIR / "faiss.index")
        shutil.copy2(META_PATH, DEST_DIR / "meta.jsonl")

        # 4. Tar up *contents* (not parent dir) into model.tar.gz
        with tarfile.open(ARCHIVE, "w:gz") as tar:
            for file_path in DEST_DIR.iterdir():
                tar.add(file_path, arcname=file_path.name)

        print(f"Created {ARCHIVE} containing {len(list(DEST_DIR.iterdir()))} files.")

    if args.question:
        print(_answer(args.question))


if __name__ == "__main__":  # pragma: no cover
    main()
