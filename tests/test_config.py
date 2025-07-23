import argparse

from vgj_chat.config import Config


def test_from_env(monkeypatch):
    monkeypatch.setenv("VGJ_TOP_K", "7")
    monkeypatch.setenv("VGJ_DEBUG", "false")
    monkeypatch.setenv("VGJ_FAISS_CUDA", "false")
    cfg = Config.from_env()
    assert cfg.top_k == 7
    assert cfg.debug is False
    assert cfg.faiss_cuda is False


def test_apply_cli_args():
    cfg = Config()
    parser = argparse.ArgumentParser()
    Config.add_argparse_args(parser)
    args = parser.parse_args(
        ["--top-k", "3", "--debug", "true", "--faiss-cuda", "false"]
    )
    new_cfg = cfg.apply_cli_args(args)
    assert new_cfg.top_k == 3
    assert new_cfg.debug is True
    assert new_cfg.faiss_cuda is False
    # unchanged value
    assert new_cfg.index_path == cfg.index_path
