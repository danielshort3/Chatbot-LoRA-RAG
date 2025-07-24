import argparse
import subprocess


STEPS = [
    ["python", "scripts/crawl.py"],
    ["python", "scripts/build_index.py"],
    ["python", "scripts/build_dataset.py"],
    ["python", "scripts/finetune.py"],
    ["python", "-m", "vgj_chat", "--compare"],
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit number of pages to crawl",
    )
    args = parser.parse_args()

    steps = STEPS.copy()
    steps[0] = ["python", "scripts/crawl.py", "--limit", str(args.limit)]

    for cmd in steps:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":  # pragma: no cover
    main()

