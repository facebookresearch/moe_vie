# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Getting started with MoE-ViE: zero-shot image classification.

Extract image and text features and compute their similarity.

Example
-------
    PYTHONPATH=src python demo/demo.py \\
        --model MoEViE-L16-384 \\
        --pretrained hf://facebook/MoEViE-L16-384:MoEViE-L16-384.pt \\
        --image path/to/cat.jpg --labels "a diagram" "a dog" "a cat"
"""

import argparse

import torch
from PIL import Image

from open_clip import create_model_and_transforms, get_tokenizer, image_to_device

PREPROCESS = {
    "MoEViE-B16-224": {"patch_size": 16, "size_range": (224, 224)},
    "MoEViE-L16-384": {"patch_size": 16, "size_range": (384, 384)},
    "MoEViE-H14-448": {"patch_size": 14, "size_range": (448, 448)},
}
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def parse_args():
    p = argparse.ArgumentParser(description="MoE-ViE zero-shot image classification demo.")
    p.add_argument("--model", default="MoEViE-L16-384", choices=list(PREPROCESS.keys()))
    p.add_argument("--pretrained", required=True, help="local .pt path, or hf://<repo>:<file> to fetch from the Hugging Face Hub")
    p.add_argument("--image", required=True, help="path to an input image")
    p.add_argument("--labels", nargs="+", default=["a diagram", "a dog", "a cat"],
                   help="candidate text labels to score against the image")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pp = PREPROCESS[args.model]

    model, _, transform = create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        force_preprocess_cfg={
            "patch_size": pp["patch_size"],
            "size_range": pp["size_range"],
            "center_crop": True,
            "window_size": 1,
        },
        image_mean=MEAN,
        image_std=STD,
    )
    model = model.to(device).eval()
    tokenizer = get_tokenizer(args.model)

    packed, _ = transform.collate_fn([(transform(Image.open(args.image).convert("RGB")), 0)])
    packed = image_to_device(packed, device, torch.float32, mean=MEAN, std=STD)
    text = tokenizer(args.labels).to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
        image_features = model.encode_image(packed, normalize=True)
        text_features = model.encode_text(text, normalize=True)
        probs = (model.logit_scale.exp() * image_features @ text_features.T).softmax(dim=-1)

    print("Label probabilities:")
    for label, prob in zip(args.labels, probs[0].tolist()):
        print(f"  {label:<20s} {prob:.4f}")


if __name__ == "__main__":
    main()
