from __future__ import annotations

import argparse

from .config import CFG, Config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VGJ Chat demo")
    Config.add_argparse_args(parser)
    args = parser.parse_args(argv)

    global CFG
    CFG = CFG.apply_cli_args(args)

    from .ui.gradio_app import build_demo

    demo = build_demo()
    demo.queue()
    demo.launch()


if __name__ == "__main__":  # pragma: no cover
    main()
