"""Scale-to-zero CPU ASGI deployment; model artifacts are baked into the image."""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("app")
    .add_local_file("scripts/download_models.py", "/root/download_models.py")
    .env({"CROPVISION_MODEL_ROOT": "/models"})
    .run_commands("python /root/download_models.py")
)

modal_service = modal.App("kisansetu-cropvision-ml")


@modal_service.function(
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
