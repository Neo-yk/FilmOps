"""Minimal loader for InternVL3 + LoRA-merged checkpoints.

FilmOps consumes InternVL **for inference only** — model definitions are
loaded via ``trust_remote_code=True`` from each checkpoint directory's own
``modeling_*.py`` files. The full InternVL training framework is therefore
*not* a runtime dependency of this package.

This loader handles two known fragility points:

1. ``trust_remote_code=True`` copies modeling files into
   ``~/.cache/huggingface/modules/transformers_modules/<ckpt_name>/``.
   If the checkpoint files are patched on disk afterwards, the cached copy
   is stale. :func:`clear_internvl_hf_cache` removes any cached
   ``checkpoint*`` modules so the latest files are picked up.
2. Some checkpoints expect their own directory to be importable as a
   sys.path entry (for relative imports between modeling files).
   :func:`load_internvl_model` injects it when needed.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import sys
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def clear_internvl_hf_cache() -> None:
    """Remove stale HF-cached copies of ``checkpoint*`` modeling modules."""
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules"
    )
    if not os.path.isdir(cache_dir):
        return
    for d in glob.glob(os.path.join(cache_dir, "checkpoint*")):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except OSError as e:
            logger.warning("Could not remove stale HF cache %s: %s", d, e)


def _ensure_path_importable(model_path: str) -> None:
    """Add ``model_path`` to ``sys.path`` if it contains local modeling files.

    Also injects the bundled ``internvl`` config shim (under
    ``filmops/backbones/``) so that checkpoint-level config files that
    ``import internvl.model.internlm2`` / ``internvl.model.phi3`` can
    resolve without a full InternVL installation.
    """
    # Inject bundled internvl shim into sys.path
    _backbones_dir = os.path.dirname(os.path.abspath(__file__))
    if _backbones_dir not in sys.path:
        sys.path.insert(0, _backbones_dir)

    if not os.path.isdir(model_path):
        return
    has_modeling_files = any(
        f.startswith("modeling_") for f in os.listdir(model_path)
    )
    if has_modeling_files and model_path not in sys.path:
        sys.path.insert(0, model_path)


def load_internvl_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[object, object]:
    """Load an InternVL3-style checkpoint and its tokenizer.

    Args:
        model_path: Directory containing the merged InternVL checkpoint
            (with its own ``modeling_*.py`` / ``config.json`` files).
        device: Target device for the model.
        torch_dtype: Loading dtype. Defaults to ``bfloat16``.

    Returns:
        Tuple of ``(model, tokenizer)``.
    """
    from transformers import AutoModel, AutoTokenizer

    clear_internvl_hf_cache()
    _ensure_path_importable(model_path)

    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    return model, tokenizer
