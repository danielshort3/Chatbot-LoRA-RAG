import argparse
import subprocess
import shutil
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

    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    shutil.copy("faiss.index", model_dir / "faiss.index")
    shutil.copy("meta.jsonl", model_dir / "meta.jsonl")
    merged_src = Path("mistral-merged-4bit")
    merged_dst = model_dir / merged_src.name
    if merged_dst.exists():
        shutil.rmtree(merged_dst)
    if merged_src.exists():
        shutil.copytree(merged_src, merged_dst)


if __name__ == "__main__":  # pragma: no cover
    main()
