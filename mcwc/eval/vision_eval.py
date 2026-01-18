from __future__ import annotations

from typing import Optional

import torch


def eval_top1_accuracy(model, dataloader, device: str = "cpu", max_batches: int = 100) -> float:
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

    return 100.0 * correct / max(total, 1)
