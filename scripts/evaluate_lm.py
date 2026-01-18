from __future__ import annotations

import argparse

import torch

from mcwc.data.lm_wikitext import load_wikitext
from mcwc.eval.lm_eval import eval_lm_perplexity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--decoded", type=str, default="", help="Path to decoded state_dict (.pt). If empty, evaluates the original model")
    ap.add_argument("--dataset", type=str, default="wikitext-2")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_batches", type=int, default=50)
    ap.add_argument("--seq_len", type=int, default=1024)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    if args.decoded:
        sd = torch.load(args.decoded, map_location="cpu")
        model.load_state_dict(sd, strict=False)

    if args.dataset == "wikitext-2":
        dsname = "wikitext-2-raw-v1"
    elif args.dataset == "wikitext-103":
        dsname = "wikitext-103-raw-v1"
    else:
        raise ValueError("dataset must be wikitext-2 or wikitext-103")

    ds = load_wikitext(dsname, split=args.split)
    ppl = eval_lm_perplexity(model, tokenizer, ds, device=args.device, max_batches=args.max_batches, seq_len=args.seq_len)
    print(f"Perplexity ({args.dataset}/{args.split}): {ppl:.4f}")


if __name__ == "__main__":
    main()
