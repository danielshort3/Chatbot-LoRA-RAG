from __future__ import annotations

import argparse
import logging

from dataclasses import replace

import vgj_chat.config as config

CFG = config.CFG
Config = config.Config


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
    config.CFG = CFG
    if CFG.compare_mode:
        logging.info("Compare mode enabled: launching dual-chat UI")

    from .ui.gradio_app import build_demo

    demo = build_demo()
    demo.queue()
    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":  # pragma: no cover
    main()
