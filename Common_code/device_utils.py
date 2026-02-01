from __future__ import annotations

import torch


def resolve_torch_device(device: str | torch.device | None) -> torch.device:
    """
    Resolve a user/device string to a concrete torch.device.

    Supported inputs:
    - None / "auto": choose best available backend in order: CUDA -> MPS -> CPU
    - "cpu"
    - "cuda" / "cuda:0" (requires torch.cuda.is_available())
    - "mps" (Apple Metal / MPS backend; requires torch.backends.mps.is_available())
    - "metal" (alias for "mps")
    - a torch.device instance
    """
    if isinstance(device, torch.device):
        return device

    d = str(device or "auto").strip().lower()
    d = "mps" if d == "metal" else d

    if d in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA device, but torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch build or use --device cpu/auto.")
        return torch.device(d)

    if d == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("Requested MPS (Apple Metal) device, but torch.backends.mps.is_available() is False. Use --device cpu/auto.")
        return torch.device("mps")

    if d == "cpu":
        return torch.device("cpu")

    # Allow advanced torch device strings (e.g., "cuda:1") to raise meaningful torch errors.
    return torch.device(d)

