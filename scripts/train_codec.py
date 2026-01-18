from __future__ import annotations

import argparse

import torch

from mcwc.codec.mcwc_codec import MCWCCodec, MCWCConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="Output bitstream path")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--qbits", type=int, default=8)
    ap.add_argument("--align", type=str, default="greedy", choices=["greedy", "hungarian"])
    ap.add_argument("--compress", type=str, default="zstd", choices=["zstd", "zlib"])
    ap.add_argument("--use_sortproj", action="store_true")
    ap.add_argument("--sortproj_seed", type=int, default=0)
    args = ap.parse_args()

    # Reference "training": MCWC in this repo fits predictors on weights directly during encode.
    # This entrypoint exists to match a paper-style pipeline.
    from transformers import AutoModelForCausalLM, AutoConfig  # type: ignore

    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    codec = MCWCCodec(MCWCConfig(
        k=args.k,
        qbits=args.qbits,
        align=args.align,
        compress=args.compress,
        use_sortproj=args.use_sortproj,
        sortproj_seed=args.sortproj_seed,
    ))

    codec.encode(model, model_id=args.model, out_path=args.out, store_config_json=cfg.to_json_string())
    print(f"[OK] Trained+encoded bitstream: {args.out}")


if __name__ == "__main__":
    main()
