"""Pinned model identifiers and conservative screening settings."""

from __future__ import annotations

import os

MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
MAX_IMAGE_SIDE = 8_000
ACCEPTED_MIME = frozenset({"image/jpeg", "image/png", "image/webp"})

GATE_MODEL_ID = "google/siglip-base-patch16-224"
# Resolved from the public Hub API on 2026-09-15; never float at repository HEAD.
GATE_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
DISEASE_MODEL_ID = "Arko007/agromind-plant-disease-mobilenet"
DISEASE_REVISION = "b7c206b2426711ee9467d3991889448313f462f1"

GATE_MIN_SHARE = 0.46
GATE_MIN_MARGIN = 0.12
# Crop naming is a second gate, so prefer an abstention over calling an onion
# leaf a supported crop from a weak zero-shot match.
CROP_MIN_SHARE = 0.60
CROP_MIN_MARGIN = 0.20
DISEASE_MIN_SHARE = 0.72
DISEASE_MIN_MARGIN = 0.20


def model_cache_root() -> str:
    return os.getenv("CROPVISION_MODEL_ROOT", "").strip()


def api_key() -> str:
    return os.getenv("KISANSETU_ML_API_KEY", "")
