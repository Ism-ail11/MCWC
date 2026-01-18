from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from mcwc.data.vision_torchvision import load_cifar100
from mcwc.eval.vision_eval import eval_top1_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="google/vit-base-patch16-224")
    ap.add_argument("--decoded", type=str, default="", help="decoded state_dict path")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_batches", type=int, default=100)
    ap.add_argument("--data_root", type=str, default="./data")
    args = ap.parse_args()

    from transformers import AutoModelForImageClassification  # type: ignore

    model = AutoModelForImageClassification.from_pretrained(args.model, torch_dtype=torch.float32)
    if args.decoded:
        sd = torch.load(args.decoded, map_location="cpu")
        model.load_state_dict(sd, strict=False)

    ds = load_cifar100(root=args.data_root, train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    acc = eval_top1_accuracy(model, dl, device=args.device, max_batches=args.max_batches)
    print(f"Top-1 accuracy (CIFAR-100): {acc:.2f}%")


if __name__ == "__main__":
    main()
