from __future__ import annotations

import argparse

from .ui.gradio_app import demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VGJ Chat demo")
    _ = parser.parse_args(argv)
    demo.queue()
    demo.launch()


if __name__ == "__main__":  # pragma: no cover
    main()
