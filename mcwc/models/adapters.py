from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class FFNSpec:
    """Defines a pair of tensors representing the FFN up and down projections."""

    layer: int
    up: str
    down: str
    family: str


def infer_ffn_specs_from_state_dict(state: Dict[str, torch.Tensor]) -> List[FFNSpec]:
    """Infer FFN tensor pairs across common HF transformer families.

    Supports:
      - GPTNeoX / Pythia: gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight and dense_4h_to_h.weight
      - OPT: model.decoder.layers.{i}.fc1.weight and fc2.weight
      - GPT-2: transformer.h.{i}.mlp.c_fc.weight and c_proj.weight
      - ViT: encoder.layer.{i}.intermediate.dense.weight and output.dense.weight

    Returns sorted specs by layer.
    """
    specs: List[FFNSpec] = []

    # GPTNeoX / Pythia
    pat_up = re.compile(r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_h_to_4h\.weight$")
    pat_dn = re.compile(r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_4h_to_h\.weight$")
    ups: Dict[int, str] = {}
    dns: Dict[int, str] = {}
    for k in state.keys():
        m = pat_up.match(k)
        if m:
            ups[int(m.group(1))] = k
        m = pat_dn.match(k)
        if m:
            dns[int(m.group(1))] = k
    for i in sorted(set(ups) & set(dns)):
        specs.append(FFNSpec(layer=i, up=ups[i], down=dns[i], family="gpt_neox"))

    # OPT
    pat_up = re.compile(r"^model\.decoder\.layers\.(\d+)\.fc1\.weight$")
    pat_dn = re.compile(r"^model\.decoder\.layers\.(\d+)\.fc2\.weight$")
    ups, dns = {}, {}
    for k in state.keys():
        m = pat_up.match(k)
        if m:
            ups[int(m.group(1))] = k
        m = pat_dn.match(k)
        if m:
            dns[int(m.group(1))] = k
    for i in sorted(set(ups) & set(dns)):
        specs.append(FFNSpec(layer=i, up=ups[i], down=dns[i], family="opt"))

    # GPT-2
    pat_up = re.compile(r"^transformer\.h\.(\d+)\.mlp\.c_fc\.weight$")
    pat_dn = re.compile(r"^transformer\.h\.(\d+)\.mlp\.c_proj\.weight$")
    ups, dns = {}, {}
    for k in state.keys():
        m = pat_up.match(k)
        if m:
            ups[int(m.group(1))] = k
        m = pat_dn.match(k)
        if m:
            dns[int(m.group(1))] = k
    for i in sorted(set(ups) & set(dns)):
        specs.append(FFNSpec(layer=i, up=ups[i], down=dns[i], family="gpt2"))

    # ViT
    pat_up = re.compile(r"^vit\.encoder\.layer\.(\d+)\.intermediate\.dense\.weight$")
    pat_dn = re.compile(r"^vit\.encoder\.layer\.(\d+)\.output\.dense\.weight$")
    ups, dns = {}, {}
    for k in state.keys():
        m = pat_up.match(k)
        if m:
            ups[int(m.group(1))] = k
        m = pat_dn.match(k)
        if m:
            dns[int(m.group(1))] = k
    for i in sorted(set(ups) & set(dns)):
        specs.append(FFNSpec(layer=i, up=ups[i], down=dns[i], family="vit"))

    # sort by layer within family (stable overall)
    specs.sort(key=lambda s: (s.family, s.layer))
    return specs
