import importlib
import subprocess
import sys
from types import ModuleType


def test_modules_import_from_wheel(tmp_path):
    # Stub heavy optional dependencies so imports succeed
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *a, **k: None
    sys.modules.setdefault("datasets", datasets)

    pandas_mod = ModuleType("pandas")
    pandas_mod.__version__ = "0.0"
    sys.modules.setdefault("pandas", pandas_mod)

    sklearn_mod = ModuleType("sklearn")
    model_selection = ModuleType("sklearn.model_selection")
    model_selection.train_test_split = lambda *a, **k: ([], [])
    sklearn_mod.model_selection = model_selection
    sys.modules.setdefault("sklearn", sklearn_mod)
    sys.modules.setdefault("sklearn.model_selection", model_selection)

    peft_mod = sys.modules.get("peft") or ModuleType("peft")
    peft_mod.LoraConfig = object
    peft_mod.get_peft_model = lambda *a, **k: None
    peft_mod.prepare_model_for_kbit_training = lambda *a, **k: None
    sys.modules["peft"] = peft_mod

    hf_hub = ModuleType("huggingface_hub")
    hf_hub.login = lambda *a, **k: None
    sys.modules.setdefault("huggingface_hub", hf_hub)

    tfm = sys.modules.get("transformers") or ModuleType("transformers")
    tfm.EarlyStoppingCallback = object
    tfm.TrainingArguments = object
    sys.modules["transformers"] = tfm

    trl_mod = ModuleType("trl")
    trl_mod.SFTTrainer = object
    sys.modules.setdefault("trl", trl_mod)

    wheel_dir = tmp_path / "wheel"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_dir),
        ],
        check=True,
    )
    wheel = next(wheel_dir.glob("vgj_chat-*.whl"))
    install_dir = tmp_path / "install"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_dir),
            str(wheel),
        ],
        check=True,
    )
    sys.path.insert(0, str(install_dir))
    try:
        importlib.import_module("vgj_chat.models.rag")
        importlib.import_module("vgj_chat.models.finetune")
    finally:
        sys.path.remove(str(install_dir))
