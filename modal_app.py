"""Scale-to-zero CPU ASGI deployment; model artifacts are baked into the image."""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    # PyPI's Linux wheel pulls CUDA components even for a CPU-only Function.
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("app", copy=True)
    .add_local_file("scripts/download_models.py", "/root/download_models.py", copy=True)
    .env({"CROPVISION_MODEL_ROOT": "/models"})
    .run_commands("python /root/download_models.py")
)

app = modal.App("kisansetu-cropvision-ml")


@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    secrets=[modal.Secret.from_name("kisansetu-ml-api-key")],
    timeout=120,
)
@modal.asgi_app()
def fastapi_service():
    from app.api import app

    return app
