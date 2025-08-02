import json
import os
from typing import Any, Tuple


def model_fn(model_dir: str) -> Any:
    """Load model artifacts from ``model_dir`` and prepare for inference."""
    os.environ.setdefault(
        "VGJ_MERGED_MODEL_DIR", os.path.join(model_dir, "mistral-merged-4bit")
    )
    os.environ.setdefault("VGJ_INDEX_PATH", os.path.join(model_dir, "faiss.index"))
    os.environ.setdefault("VGJ_META_PATH", os.path.join(model_dir, "meta.jsonl"))
    from vgj_chat.models import rag

    return rag


def input_fn(request_body: bytes, request_content_type: str) -> str:
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return data.get("question") or data.get("inputs") or ""
    if request_content_type == "text/plain":
        return request_body.decode("utf-8")
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: str, model: Any) -> str:
    return model.chat(input_data)


def output_fn(prediction: str, accept: str) -> Tuple[str, str]:
    if accept == "application/json":
        return json.dumps({"answer": prediction}), accept
    return prediction, "text/plain"
