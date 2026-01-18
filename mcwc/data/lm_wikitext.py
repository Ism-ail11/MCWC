from __future__ import annotations

from typing import Dict, Iterator, List, Optional


def load_wikitext(dataset: str = "wikitext-2-raw-v1", split: str = "test"):
    """Load WikiText with HF datasets. Requires `datasets`.

    dataset: "wikitext-2-raw-v1" or "wikitext-103-raw-v1".
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("wikitext", dataset, split=split)
    return ds
