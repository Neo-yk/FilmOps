"""Backbone loaders shared across operators.

These thin wrappers exist so that operator code stays free of model-loading
boilerplate (HF cache cleanup, ``trust_remote_code`` flags, torch.hub
invocations, etc.).
"""

from filmops.backbones.internvl_loader import (
    clear_internvl_hf_cache,
    load_internvl_model,
)

__all__ = ["clear_internvl_hf_cache", "load_internvl_model"]
