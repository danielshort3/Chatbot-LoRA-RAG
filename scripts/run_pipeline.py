import argparse
import subprocess
import tarfile
from pathlib import Path

STEPS = [
    ["python", "scripts/crawl.py"],
    ["python", "scripts/build_index.py"],
    ["python", "scripts/build_dataset.py"],
    ["python", "scripts/finetune.py"],
    ["python", "scripts/merge_lora.py"],
    ["python", "-m", "vgj_chat", "--compare"],
]


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

    steps = STEPS.copy()
    if args.limit is not None:
        steps[0] = ["python", "scripts/crawl.py", "--limit", str(args.limit)]
    if not args.launch_chatbot:
        steps.pop()

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
