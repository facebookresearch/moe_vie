# MoE-ViE: Mixture-of-Experts Vision Encoders

![ECCV 2026](https://img.shields.io/badge/ECCV-2026-1b3d6d.svg)
[![Paper](https://img.shields.io/badge/Paper-arXiv:2608.17402-b31b1b.svg)](https://arxiv.org/abs/2608.17402)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-MoE--ViE-FFD21E.svg)](https://huggingface.co/models?search=facebook/MoEViE)
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](LICENSE)

Official code release for ECCV 2026 paper **MoE-ViE: Mixture of Experts Vision Encoder for
Efficient Image and Video Understanding**. This repository provides the model definition, config files, and a
reproducible zero-shot evaluation suite for image/video classification and
image/video–text retrieval.

MoE-ViE uses a robust contrastive pretraining recipe with a Mixture-of-Experts
vision transformer executed with custom [Triton](https://github.com/openai/triton) kernels.
It comes in three sizes, **B / L / H**, that are competitive with strong CLIP baselines across
classification and retrieval.

## Model performance

Top-1 accuracy (%) for classification, recall@1 (%) for retrieval.

| Scale | Checkpoint | ImageNet&#8209;1k | ObjectNet | COCO&nbsp;T2I | Kinetics&#8209;400 | VTT&nbsp;T2V |
|:-----------:|:--------------:|:-----------:|:---------:|:--------:|:------------:|:------------:|
| **B/16**&nbsp;224px | [🤗&nbsp;MoEViE‑B16‑224](https://huggingface.co/facebook/MoEViE-B16-224) | 79.3 | 74.4 | 52.1 | 68.3 | 47.9 |
| **L/16**&nbsp;384px | [🤗&nbsp;MoEViE‑L16‑384](https://huggingface.co/facebook/MoEViE-L16-384) | 83.6 | 85.0 | 57.2 | 74.5 | 50.5 |
| **H/14**&nbsp;448px | [🤗&nbsp;MoEViE‑H14‑448](https://huggingface.co/facebook/MoEViE-H14-448) | 85.1 | 87.0 | 56.8 | 76.9 | 51.6 |

## Installation

```bash
git clone <repo-url> && cd moe_vie
pip install -r requirements.txt
```

Requirements: Python ≥ 3.10 and a **CUDA GPU** — the Mixture-of-Experts kernels
are compiled with Triton at runtime. Pin `torch` / `triton` to compatible
versions.

## Usage

### Quick start

Zero-shot classify an image against candidate text labels:

```bash
PYTHONPATH=src python demo/demo.py \
    --model MoEViE-L16-384 --pretrained hf://facebook/MoEViE-L16-384:MoEViE-L16-384.pt \
    --image path/to/image.jpg --labels "a diagram" "a dog" "a cat"
```

See [`demo/demo.py`](demo/demo.py) for the image/text feature-extraction pattern.

### Data preparation

**WebDataset benchmarks** (image classification & image–text retrieval) are read
in the [WebDataset](https://github.com/webdataset/webdataset) `.tar` shard format
used by [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark). Point
`--dataset_root` at the directory that holds them, e.g.
`--dataset_root "/path/to/wds/{dataset_cleaned}/"`.

**Video benchmarks** (video classification & retrieval) are read from a single
root set via `CLIP_BENCHMARK_DATA_ROOT` (default: `datasets`).

### Zero-shot evaluation

```bash
model='MoEViE-H14-448'
DATASETS="wds/wds_imagenet1k"
DATA_ROOT="/path/to/wds/{dataset_cleaned}/"
export PYTHONPATH=src
python -m clip_benchmark.cli eval \
    --model $model \
    --pretrained $CHECKPOINT \
    --dataset $DATASETS \
    --dataset_root $DATA_ROOT \
    --output "results/{pretrained}_{dataset}_{model}_{task}.json" \
    --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5 \
    --force-preprocess-cfg patch_size=$PATCH_SIZE window_size=1 "size_range=($SIZE,$SIZE)" \
    --batch_size $BATCH_SIZE --num_workers $WORKERS \
    --use_optimized_inference   # optional
```

## Repository Layout

```
src/open_clip/        # model definition, MoE transformer, Triton kernels, model_configs/
src/clip_benchmark/   # zero-shot evaluation (datasets, metrics, CLI)
```

## License

Released under [LICENSE](LICENSE). Model cards
are provided on the Hugging Face model pages.

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2026moevie,
  title={MoE-ViE: Mixture of Experts Vision Encoder for Efficient Image and Video Understanding},
  author={Bonan Zhang and Shiyu Dong and Quan Hung Tran and Katharina Gschwind and Shuqi Yang and Sijia Chen and Adel Ahmadyan and Seungwhan Moon and Lu Zhang and Ahmed Kirmani and Babak Damavandi and Anuj Kumar},
  journal={arXiv preprint arXiv:2608.17402},
  year={2026}
}
```

## Acknowledgements

This code builds on [OpenCLIP](https://github.com/mlfoundations/open_clip) and
[CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark);
the BPE tokenizer originates from [OpenAI CLIP](https://github.com/openai/CLIP).
We thank the authors of these projects.
