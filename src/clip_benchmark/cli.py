# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
from copy import copy
from itertools import product
import ast
import pathlib

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import torch

from clip_benchmark.datasets.builder import (build_dataset,
                                             get_dataset_collate_fn,
                                             get_dataset_default_task,
                                             is_video_dataset,)
from clip_benchmark.metrics import (zeroshot_classification,
                                    zeroshot_retrieval)
import open_clip

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = getattr(namespace, self.dest, {})
        for value in values:
            key, value = value.split("=")
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)
        setattr(namespace, self.dest, kw)

def get_parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_eval = subparsers.add_parser('eval', help='Evaluate')
    parser_eval.add_argument('--dataset', type=str, default="cifar10", nargs="+")
    parser_eval.add_argument('--dataset_root', default="root", type=str)
    parser_eval.add_argument('--split', type=str, default="test")
    parser_eval.add_argument('--model', type=str, nargs="+", default=["ViT-B-32-quickgelu"])
    parser_eval.add_argument('--pretrained', type=str, nargs="+", default=["laion400m_e32"])
    parser_eval.add_argument('--task', type=str, default="auto", choices=["zeroshot_classification", "zeroshot_retrieval", "auto"])
    parser_eval.add_argument('--no_amp', action="store_false", dest="amp", default=True)
    parser_eval.add_argument('--num_workers', default=12, type=int)
    parser_eval.add_argument('--recall_k', default=[1, 5, 10], type=int, nargs="+")
    parser_eval.add_argument('--seed', default=0, type=int)
    parser_eval.add_argument('--batch_size', default=64, type=int)
    parser_eval.add_argument("--image-mean", type=float, nargs="+", default=None, metavar="MEAN")
    parser_eval.add_argument("--image-std", type=float, nargs="+", default=None, metavar="STD")
    parser_eval.add_argument("--force-preprocess-cfg", nargs="*", default={}, action=ParseKwargs)
    parser_eval.add_argument("--force-vision-cfg", nargs="*", default={}, action=ParseKwargs)
    parser_eval.add_argument("--force-text-cfg", nargs="*", default={}, action=ParseKwargs)
    parser_eval.add_argument('--language', default="en", type=str, nargs="+")
    parser_eval.add_argument('--output', default="result.json", type=str)
    parser_eval.add_argument('--quiet', dest='verbose', action="store_false")
    parser_eval.add_argument('--skip_existing', default=False, action="store_true")
    parser_eval.add_argument('--model_type', default="open_clip", type=str, choices=["open_clip"])
    parser_eval.add_argument('--num_frames', default=4, type=int)
    parser_eval.add_argument('--use_optimized_inference', default=False, action='store_true')
    parser_eval.add_argument('--reweight_retrieval', default=True, action=argparse.BooleanOptionalAction)
    parser_eval.set_defaults(which='eval')

    args = parser.parse_args()
    return parser, args

def main():
    parser, base = get_parser_args()
    if not hasattr(base, "which"):
        parser.print_help()
        return
    if base.which == "eval":
        main_eval(base)

def main_eval(base):
    models = list(product(base.model, base.pretrained))

    datasets = []
    for name in _as_list(base.dataset):
        if os.path.isfile(name):
            datasets.extend([l.strip() for l in open(name).readlines() if l.strip()])
        else:
            datasets.append(name)

    languages = _as_list(base.language)

    if base.verbose:
        print(f"Models: {models}")
        print(f"Datasets: {datasets}")
        print(f"Languages: {languages}")
    runs = product(models, datasets, languages)
    for (model, pretrained), (dataset), (language) in runs:
        args = copy(base)
        args.model = model
        args.pretrained = pretrained
        args.dataset = dataset
        args.language = language
        run(args)

def _as_list(l):
    if not l:
        return []
    return [l] if type(l) != list else l

def get_basename_and_parent_folder(path):
    p = pathlib.Path(path)
    parent_folder = p.parents[1].name if len(p.parents) >= 2 else ""
    basename = p.stem
    return f"{parent_folder}-{basename}"

def run(args):
    args.device = "cuda"
    torch.manual_seed(args.seed)
    task = args.task
    if args.dataset.startswith("wds/"):
        dataset_name = args.dataset.replace("wds/", "", 1)
    else:
        dataset_name = args.dataset
    if task == "auto":
        task = get_dataset_default_task(dataset_name)
    pretrained_slug = get_basename_and_parent_folder(args.pretrained)
    dataset_slug = dataset_name.replace('/', '_')
    output = args.output.format(
        model=args.model,
        pretrained=pretrained_slug,
        task=task,
        dataset=dataset_slug,
        language=args.language
    )
    if os.path.exists(output) and args.skip_existing:
        if args.verbose:
            print(f"Skip {output}, exists already.")
        return
    if args.verbose:
        print(f"Running '{task}' on '{dataset_name}' with the model '{args.pretrained}' on language '{args.language}'")
    dataset_root = args.dataset_root.format(dataset=dataset_name, dataset_cleaned=dataset_name.replace("/", "-"))
    model, _, transform = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        force_preprocess_cfg=args.force_preprocess_cfg,
        force_vision_cfg=args.force_vision_cfg,
        force_text_cfg=args.force_text_cfg,
        use_optimized_inference=args.use_optimized_inference,
        image_mean=args.image_mean,
        image_std=args.image_std,
    )
    model = model.to(args.device)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()
    dataset = build_dataset(
        dataset_name=args.dataset,
        root=dataset_root,
        transform=transform,
        split=args.split,
        download=True,
        language=args.language,
        task=task,
        num_frames=args.num_frames
    )
    collate_fn = getattr(transform, "collate_fn", None)
    if collate_fn is None:
        collate_fn = get_dataset_collate_fn(args.dataset)

    if args.verbose:
        try:
            print(f"Dataset size: {len(dataset)}")
        except TypeError:
            print("IterableDataset has no len()")
        print(f"Dataset split: {args.split}")
        if hasattr(dataset, "classes") and dataset.classes:
            try:
                print(f"Dataset classes: {dataset.classes}")
                print(f"Dataset number of classes: {len(dataset.classes)}")
            except AttributeError:
                print("Dataset has no classes.")

    if args.dataset.startswith("wds/"):
        dataloader = torch.utils.data.DataLoader(
            dataset.batched(args.batch_size, collation_fn=collate_fn), batch_size=None,
            shuffle=False, num_workers=args.num_workers,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_fn
        )
    if task == "zeroshot_classification":
        zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
        if args.verbose:
            print(f"Zero-shot templates: {zeroshot_templates}")
        classnames = dataset.classes if hasattr(dataset, "classes") else None
        assert (zeroshot_templates is not None and classnames is not None), "Dataset does not support classification"
        metrics = zeroshot_classification.evaluate(
            model,
            dataloader,
            tokenizer,
            classnames, zeroshot_templates,
            video_dataset=is_video_dataset(args.dataset),
            device=args.device,
            amp=args.amp,
            verbose=args.verbose,
            args=args,
        )
    elif task == "zeroshot_retrieval":
        metrics = zeroshot_retrieval.evaluate(
            model,
            dataloader,
            tokenizer,
            video_dataset=is_video_dataset(args.dataset),
            recall_k_list=args.recall_k,
            device=args.device,
            amp=args.amp,
            args=args,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    dump = {
        "dataset": args.dataset,
        "model": args.model,
        "pretrained": args.pretrained,
        "task": task,
        "metrics": metrics,
        "language": args.language,
    }
    if args.verbose:
        print(f"Dump results to: {output}")
    with open(output, "w") as f:
        json.dump(dump, f)
    return 0

if __name__ == "__main__":
    sys.exit(main())

