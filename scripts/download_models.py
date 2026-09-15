"""Preload only required pinned checkpoint artifacts during a Modal image build."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from app.config import DISEASE_MODEL_ID, DISEASE_REVISION, GATE_MODEL_ID, GATE_REVISION


def main() -> None:
    root = Path(os.getenv("CROPVISION_MODEL_ROOT", "/models"))
    snapshot_download(
        repo_id=GATE_MODEL_ID,
        revision=GATE_REVISION,
        local_dir=root / "gate",
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
    )
    snapshot_download(
        repo_id=DISEASE_MODEL_ID,
        revision=DISEASE_REVISION,
        local_dir=root / "disease",
        allow_patterns=["labels.txt", "newplant_model_final.pth"],
    )
    if not (root / "gate" / "model.safetensors").is_file():
        raise RuntimeError("Gate checkpoint download was incomplete.")
    if not (root / "disease" / "newplant_model_final.pth").is_file():
        raise RuntimeError("Disease checkpoint download was incomplete.")


if __name__ == "__main__":
    main()
