from __future__ import annotations

import argparse
import logging

from dataclasses import replace

from .config import CFG, Config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VGJ Chat demo")
    Config.add_argparse_args(parser)
    parser.add_argument(
        "-c",
        "--compare",
        action="store_true",
        help="Launch UI in Compare mode",
    )
    args = parser.parse_args(argv)

    global CFG
    CFG = CFG.apply_cli_args(args)
    CFG = replace(CFG, compare_mode=args.compare)
    if CFG.compare_mode:
        logging.info("Compare mode enabled: launching dual-chat UI")

    from .ui.gradio_app import build_demo

    demo = build_demo()
    demo.queue()
    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":  # pragma: no cover
    main()
