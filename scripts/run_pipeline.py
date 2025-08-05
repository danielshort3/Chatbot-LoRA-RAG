import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

CRAWL_TXT_DIR = Path("data/html_txt")
AUTO_QA_JL = Path("data/dataset/vgj_auto_dataset.jsonl")
LORA_DIR = Path("data/lora-vgj-checkpoint")

# ──────────────────────────────────────────────────────────
# NEW CONSTANTS
MERGED_SRC  = Path("data/mistral-merged-4bit")
DEST_DIR    = Path("model")          # or Path("models") if you prefer
ARCHIVE     = Path("model.tar.gz")  # will live in project root
# ──────────────────────────────────────────────────────────


def main() -> None:
    # ---------- existing CLI and pipeline code ----------
    parser = argparse.ArgumentParser(description="Run full pipeline")

    def str2bool(v: str) -> bool:
        if v.lower() in {"1", "true", "t", "yes", "y"}:
            return True
        if v.lower() in {"0", "false", "f", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError("boolean value expected")

    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of pages to crawl"
    )
    parser.add_argument(
        "--launch-chatbot",
        type=str2bool,
        default=True,
        help="Launch the chat UI after finishing the pipeline (true/false)",
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

    steps.extend(
        [
            ["python", "scripts/build_index.py"],
            ["python", "scripts/build_dataset.py"],
        ]
    )

    if (LORA_DIR / "adapter_model.safetensors").exists():
        print(f"{LORA_DIR} exists; skipping fine-tune step")
    else:
        steps.append(["python", "scripts/finetune.py", "--data", str(AUTO_QA_JL)])

    if (MERGED_SRC / "model.safetensors").exists():
        print(f"{MERGED_SRC} exists; skipping merge step")
    else:
        steps.append(["python", "scripts/merge_lora.py"])

    if args.launch_chatbot:
        steps.append(["python", "-m", "vgj_chat", "--compare"])

    for cmd in steps:
        subprocess.run(cmd, check=True)

    # ---------- NEW FILE-GATHERING LOGIC ----------
    # 1. Prep clean DEST_DIR
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Copy merged model files into DEST_DIR
    if MERGED_SRC.exists():
        for src_path in MERGED_SRC.iterdir():
            shutil.copy2(src_path, DEST_DIR / src_path.name)
    else:
        raise FileNotFoundError(f"{MERGED_SRC} not found – make sure merge_lora.py completed.")

    # 3. Copy index & metadata
    shutil.copy2("data/faiss.index", DEST_DIR / "faiss.index")
    shutil.copy2("data/meta.jsonl",  DEST_DIR / "meta.jsonl")

    # 4. Tar up *contents* (not parent dir) into model.tar.gz
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        for file_path in DEST_DIR.iterdir():
            tar.add(file_path, arcname=file_path.name)

    print(f"Created {ARCHIVE} containing {len(list(DEST_DIR.iterdir()))} files.")

if __name__ == "__main__":  # pragma: no cover
    main()
