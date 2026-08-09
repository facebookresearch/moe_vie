# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .model import (
    CLIP,
    get_cast_dtype,
    set_model_preprocess_cfg,
)
from .tokenizer import DEFAULT_CONTEXT_LENGTH, SimpleTokenizer
from .transform import (
    image_transform_v2,
    merge_preprocess_dict,
    merge_preprocess_kwargs,
    PreprocessCfg,
)

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]

def _rescan_model_configs():
    global _MODEL_CONFIGS
    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))
    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "vision_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg
    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }

_rescan_model_configs()

def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    return None

def get_tokenizer(model_name: str = "", context_length: Optional[int] = None, **kwargs):
    config = get_model_config(model_name)
    assert config is not None, f"No valid model config found for {model_name}."

    text_config = config.get("text_cfg", {})
    tokenizer_kwargs = dict(text_config.get("tokenizer_kwargs", {}), **kwargs)
    if context_length is None:
        context_length = text_config.get("context_length", DEFAULT_CONTEXT_LENGTH)

    return SimpleTokenizer(context_length=context_length, **tokenizer_kwargs)

HF_ORG = "facebook"

def fetch_checkpoint(path: str):
    """Resolve a checkpoint location to a local file.

    Accepts a local path, or ``hf://<repo_id>:<filename>`` to pull from the
    Hugging Face Hub, e.g. ``hf://facebook/MoEViE-H14-448:MoEViE-H14-448.pt``.
    """
    if path.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        repo, filename = path[len("hf://"):].split(":")
        return hf_hub_download(repo_id=repo, filename=filename)
    return path

def load_state_dict(checkpoint_path: str, map_location="cpu"):
    checkpoint_path = fetch_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint(model, checkpoint_path, strict=False):
    state_dict = load_state_dict(checkpoint_path)
    position_id_key = "text.transformer.embeddings.position_ids"
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

def _apply_post_optimizations(model, use_optimized_inference):
    if use_optimized_inference:
        from .transformer import MoEKernelWrapper
        for name, module in model.named_modules():
            if isinstance(module, MoEKernelWrapper):
                module.pad_weights()
                module.use_optimized_swiglu_inference = True

def create_model(
    model_name: str,
    pretrained: Optional[Union[str, bool]] = None,
    precision: str = "fp32",
    device: Union[str, torch.device] = "cpu",
    use_optimized_inference: bool = False,
    force_preprocess_cfg: Optional[Dict[str, Any]] = None,
    force_vision_cfg: Optional[Dict[str, Any]] = None,
    force_text_cfg: Optional[Dict[str, Any]] = None,
    **model_kwargs,
):
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    model_name = model_name.replace("/", "-")
    model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = model_cfg or get_model_config(model_name)
    if model_cfg is None:
        raise RuntimeError(f"Model config for {model_name} not found.")

    if force_vision_cfg is not None:
        model_cfg["vision_cfg"].update(force_vision_cfg)
    if force_text_cfg is not None:
        model_cfg["text_cfg"].update(force_text_cfg)

    cast_dtype = get_cast_dtype(precision)
    model_cfg = dict(model_cfg, **model_kwargs)

    model = CLIP(**model_cfg, cast_dtype=cast_dtype)

    model.to(device=device)

    if pretrained is True:
        pretrained = f"hf://{HF_ORG}/{model_name}:{model_name}.pt"

    load_checkpoint(model, pretrained)

    force_preprocess_cfg["size"] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    if use_optimized_inference:
        _apply_post_optimizations(model, use_optimized_inference)

    return model

def create_model_and_transforms(
    model_name: str,
    pretrained: Optional[Union[str, bool]] = None,
    precision: str = "fp32",
    device: Union[str, torch.device] = "cpu",
    use_optimized_inference: bool = False,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    force_preprocess_cfg: Optional[Dict[str, Any]] = None,
    **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        force_preprocess_cfg,
        mean=image_mean, std=image_std,
    )
    model = create_model(
        model_name, pretrained,
        precision=precision, device=device,
        force_preprocess_cfg=force_preprocess_cfg,
        use_optimized_inference=use_optimized_inference,
        **model_kwargs,
    )
    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)
    preprocess_val = image_transform_v2(pp_cfg)
    return model, None, preprocess_val
