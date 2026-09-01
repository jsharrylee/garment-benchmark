from __future__ import annotations

from pathlib import Path

from .model import build_model


def load_repair_model(path: Path, device: str):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.repair_config = payload["model_config"]
    model.training_metrics = payload.get("metrics", {})
    model.to(device).eval()
    return model, payload
