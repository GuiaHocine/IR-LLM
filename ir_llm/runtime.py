"""Explicit model loading and device selection, with no import-time downloads."""


def get_best_device():
    """Prefer CUDA, then Apple MPS, otherwise CPU."""
    return select_device("auto")


def select_device(name="auto"):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    return device


def load_generator(model_name, device):
    """Load a causal language model; use float32 on CPU for compatibility."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
    ).to(device)
    model.eval()
    return tokenizer, model


def set_seed(seed):
    """Seed Python and Torch; hardware kernels may still be nondeterministic."""
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
