import argparse
import subprocess
import tarfile
from pathlib import Path

CRAWL_TXT_DIR = Path("data/html_txt")
AUTO_QA_JL = Path("data/dataset/vgj_auto_dataset.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline")

    def str2bool(v: str) -> bool:
        if v.lower() in {"1", "true", "t", "yes", "y"}:
            return True
        if v.lower() in {"0", "false", "f", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError("boolean value expected")

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of pages to crawl",
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
            ["python", "scripts/finetune.py", "--data", str(AUTO_QA_JL)],
            ["python", "scripts/merge_lora.py"],
        ]
    )
    if args.launch_chatbot:
        steps.append(["python", "-m", "vgj_chat", "--compare"])

    for cmd in steps:
        subprocess.run(cmd, check=True)

    merged_src = Path("data/mistral-merged-4bit")
    archive = Path("data/mistral-rag.tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        if merged_src.exists():
            tar.add(merged_src, arcname=merged_src.name)
        tar.add("data/faiss.index", arcname="faiss.index")
        tar.add("data/meta.jsonl", arcname="meta.jsonl")
    print(f"Created {archive}")


if __name__ == "__main__":  # pragma: no cover
    main()
