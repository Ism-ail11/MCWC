from __future__ import annotations

import math
from typing import Optional

import torch


def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(loss))


def eval_lm_perplexity(model, tokenizer, dataset, device: str = "cpu", max_batches: int = 50, seq_len: int = 1024) -> float:
    """Compute approximate perplexity on a text dataset (HF datasets style).

    dataset expected to have a `text` field.
    """
    model.eval()
    model.to(device)

    losses = []
    with torch.no_grad():
        for i, ex in enumerate(dataset):
            if i >= max_batches:
                break
            text = ex.get("text", "")
            if not text:
                continue
            toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            input_ids = toks["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue
            out = model(input_ids=input_ids, labels=input_ids)
            loss = float(out.loss.detach().cpu().item())
            losses.append(loss)

    if not losses:
        return float("inf")

    mean_loss = sum(losses) / len(losses)
    return perplexity_from_loss(mean_loss)
