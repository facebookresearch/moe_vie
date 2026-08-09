# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
from .transformer import to_2tuple

@dataclass
class PreprocessCfg:
    size: Union[int, Tuple[int, int]] = 224
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN
    std: Tuple[float, ...] = OPENAI_DATASET_STD
    interpolation: str = "bicubic"
    window_size: int = None
    patch_size: int = None
    size_range: Tuple[int, int] = (224, 448)
    center_crop: bool = False

_PREPROCESS_KEYS = set(asdict(PreprocessCfg()).keys())

def merge_preprocess_dict(base: Dict, overlay: Dict):
    base_clean = {k: v for k, v in base.items() if k in _PREPROCESS_KEYS}
    if overlay:
        base_clean.update({k: v for k, v in overlay.items() if k in _PREPROCESS_KEYS and v is not None})
    return base_clean

def merge_preprocess_kwargs(base, **kwargs):
    return merge_preprocess_dict(base or {}, kwargs)

def _convert_to_rgb(image):
    return image.convert("RGB")

class PackWindowsAndPad:
    def __init__(self, window_size, patch_size, insert_cls_token=False):
        self.window_size = window_size
        self.patch_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)

    def __call__(self, img: torch.Tensor):
        c, ih, iw = img.shape
        patch_h, patch_w = self.patch_size
        h = int(math.ceil(ih / patch_h)) * patch_h
        w = int(math.ceil(iw / patch_w)) * patch_w
        pt, pb = (h - ih) // 2, (h - ih) - (h - ih) // 2
        pl, pr = (w - iw) // 2, (w - iw) - (w - iw) // 2
        if not (pt == pb and pl == pr and pr == 0):
            img = F.pad(img, [pl, pt, pr, pb], fill=0)
        img = img.unfold(-2, patch_h, patch_h).unfold(-2, patch_w, patch_w)
        img_idx = torch.arange(h * w // (patch_h * patch_w), dtype=torch.int32).reshape(h // patch_h, w // patch_w)
        img = img.reshape(c, -1, 1, *self.patch_size)
        img_idx = img_idx.reshape(-1, 1)
        idx_h, idx_w = h // self.patch_size[0], w // self.patch_size[1]
        packed_img_idx = torch.empty(img_idx.shape[0], img_idx.shape[1], PACKED.NUM_METADATA - 1, dtype=torch.int32)
        packed_img_idx[:, :, PACKED.Z].fill_(0)
        packed_img_idx[:, :, PACKED.Y] = img_idx // idx_w
        packed_img_idx[:, :, PACKED.X] = img_idx % idx_w
        packed_img_idx[:, :, PACKED.TIME].fill_(1)
        packed_img_idx[:, :, PACKED.HEIGHT].fill_(idx_h)
        packed_img_idx[:, :, PACKED.WIDTH].fill_(idx_w)
        packed_img_idx[:, :, PACKED.IDX] = img_idx
        return img, packed_img_idx

    def __repr__(self):
        return f'{self.__class__.__name__}(window_size={self.window_size}, patch_size={self.patch_size}, window_shape=none)'

class CollatePackedWindows:
    def _process_frame(self, batch):
        batch = [x for x in batch if x is not None]
        batch.sort(key=lambda x: x[0][0].shape[1])
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        if isinstance(target, (list, tuple)) and isinstance(target[0], torch.Tensor):
            target = torch.stack(target, dim=0)
        elif isinstance(target, (list, tuple)) and isinstance(target[0], list) and isinstance(target[0][0], str):
            pass
        else:
            target = torch.Tensor(target)
        num_windows = torch.Tensor([x[0].shape[1] for x in data]).long()
        packed_num_windows, packed_counts = torch.unique(num_windows, return_counts=True)
        packed_end_idx = (packed_counts * packed_num_windows).cumsum(dim=0)
        packing_boundaries = [packed_num_windows.tolist(), packed_end_idx.tolist()]
        packed_img = torch.cat([x[0] for x in data], dim=1).permute(1, 0, 2, 3, 4).contiguous()
        packed_img_idx = torch.cat([x[1] for x in data], dim=0)
        element_idx = torch.Tensor(sum([[i]*n for i, n in enumerate(num_windows)], [])).to(torch.int32)
        element_idx = element_idx.view(-1, 1).tile((1, packed_img_idx.shape[1]))
        packed_img_idx = torch.cat([packed_img_idx, element_idx[..., None]], dim=-1)
        return packed_img, packed_img_idx, num_windows.tolist(), packing_boundaries, target

    def __call__(self, batch):
        if isinstance(batch[0][0][0], (list, tuple)):
            frame_results = []
            for frame_idx in range(len(batch[0][0])):
                frame_batch = [(video[frame_idx], tgt) for video, tgt in batch]
                packed_img, packed_img_idx, num_windows, packing_boundaries, frame_target = self._process_frame(frame_batch)
                frame_results.append([(packed_img, packed_img_idx, num_windows, packing_boundaries), frame_target])
            return [fr[0] for fr in frame_results], frame_results[0][1]
        else:
            packed_img, packed_img_idx, num_windows, packing_boundaries, target = self._process_frame(batch)
            return [(packed_img, packed_img_idx, num_windows, packing_boundaries), target]

class PACKED:
    Z = 0; Y = 1; X = 2; TIME = 3; HEIGHT = 4; WIDTH = 5; IDX = 6; BATCH_IDX = 7
    NUM_METADATA = 8; ID_CLS_TOKEN = -2; ID_PAD_TOKEN = -1

def image_to_device(images, device, input_dtype, mean=None, std=None):
    if isinstance(images, torch.Tensor):
        x = images
        if 'float' in str(x.dtype):
            return x.to(device=device, dtype=input_dtype, non_blocking=True)
        elif 'uint8' in str(x.dtype):
            mean = mean or OPENAI_DATASET_MEAN
            std = std or OPENAI_DATASET_STD
            x = x.to(device, non_blocking=True).to(input_dtype or torch.float32, non_blocking=True)
            bcast = [1, -1] + [1 for _ in range(len(x.shape) - 2)]
            x.sub_(torch.as_tensor([c*255 for c in mean], dtype=x.dtype, device=x.device).view(bcast))
            x.div_(torch.as_tensor([c*255 for c in std], dtype=x.dtype, device=x.device).view(bcast))
            return x
        return x.to(device=device, non_blocking=True)
    elif isinstance(images, (list, tuple)):
        return images.__class__(image_to_device(x, device, input_dtype, mean, std) for x in images)
    return images

def image_transform(image_size, mean=None, std=None, interpolation=None,
                    window_size=None, patch_size=None, size_range=None,
                    center_crop=False, **kwargs):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)): mean = (mean,) * 3
    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)): std = (std,) * 3
    interpolation_mode = InterpolationMode.BILINEAR if (interpolation or "bicubic") == "bilinear" else InterpolationMode.BICUBIC
    normalize = Normalize(mean=mean, std=std)

    if center_crop:
        transforms = [Resize(size_range[1], interpolation=interpolation_mode), CenterCrop(size_range[1])]
    else:
        transforms = [Resize(size_range, interpolation=interpolation_mode)]
    transforms.extend([_convert_to_rgb, ToTensor(), normalize, PackWindowsAndPad(window_size, patch_size)])

    transform = Compose(transforms)
    transform.collate_fn = CollatePackedWindows()
    return transform

def image_transform_v2(cfg: PreprocessCfg, **kwargs):
    transform = image_transform(
        image_size=cfg.size, mean=cfg.mean, std=cfg.std,
        interpolation=cfg.interpolation, window_size=cfg.window_size,
        patch_size=cfg.patch_size, size_range=cfg.size_range, center_crop=cfg.center_crop,
    )
    transform.cfg = cfg
    return transform
