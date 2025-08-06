from types import SimpleNamespace

from scripts import run_pipeline
from vgj_chat.models.rag.retrieval import SentenceWindowRetriever
from vgj_chat.utils.text import token_len


def test_windowize_stride_one():
    text = "A. B. C. D."
    windows = SentenceWindowRetriever._windows_from_doc(text)
    assert windows == [(0, "A. B. C."), (0, "B. C. D.")]


def test_two_stage_retrieval_respects_limits(set_retrieval_env):
    texts = [
        "alpha one. alpha two. alpha three. alpha four.",
        "beta one. beta two. beta three. beta four.",
        "alpha five. alpha six. alpha seven. alpha eight.",
    ]
    set_retrieval_env(texts)
    retriever = SentenceWindowRetriever(doc_top_k=2, win_top_k=2)
    blocks = retriever.retrieve_windows("alpha")
    assert len(blocks) == 2
    assert all("<DOC_ID:0>" in b or "<DOC_ID:2>" in b for b in blocks)


def test_mmr_drops_redundant_windows(set_retrieval_env):
    texts = ["alpha one. alpha two. alpha three. alpha four. alpha five."]
    set_retrieval_env(texts)
    retriever = SentenceWindowRetriever(doc_top_k=1, win_top_k=3, mmr_lambda=0.3)
    blocks = retriever.retrieve_windows("alpha")
    assert len(blocks) == 1


def test_context_assembly_limits(monkeypatch):
    blocks = [
        "<DOC_ID:0> <PARA_ID:0> <URL:u0> <DATE:unknown>\na b",
        "<DOC_ID:1> <PARA_ID:0> <URL:u1> <DATE:unknown>\nc d",
        "<DOC_ID:2> <PARA_ID:0> <URL:u2> <DATE:unknown>\ne f",
    ]
    monkeypatch.setattr(
        run_pipeline.SentenceWindowRetriever, "retrieve_windows", lambda self, q: blocks
    )
    monkeypatch.setattr(run_pipeline, "CTX_TOK_LIMIT", 10)
    monkeypatch.setattr(run_pipeline, "OUT_TOK_LIMIT", 3)

    from vgj_chat.models.rag import boot as _boot

    monkeypatch.setattr(_boot, "_ensure_boot", lambda: None)

    class DummyTokOut(dict):
        def to(self, device):
            return self

    class DummyModel:
        device = "cpu"

        def generate(self, input_ids, attention_mask, max_new_tokens, do_sample):
            self.max_new_tokens = max_new_tokens
            return [list(range(len(input_ids) + max_new_tokens))]

    class DummyTokenizer:
        def __init__(self):
            self.model = None

        def __call__(self, prompt, return_tensors="pt"):
            self.last_prompt = prompt
            self.input_len = len(prompt.split())
            return DummyTokOut({"input_ids": [0] * self.input_len})

        def decode(self, tokens, skip_special_tokens=True):
            return self.last_prompt + " " + "ans " * self.model.max_new_tokens

    tok = DummyTokenizer()
    model = DummyModel()
    tok.model = model
    chat = SimpleNamespace(tokenizer=tok, model=model)
    monkeypatch.setattr(_boot, "CHAT", chat)
    monkeypatch.setattr(_boot, "CFG", SimpleNamespace(max_new_tokens=10))

    answer = run_pipeline._answer("q")
    assert blocks[0] in tok.last_prompt
    assert blocks[1] not in tok.last_prompt
    assert token_len(answer) == 3
