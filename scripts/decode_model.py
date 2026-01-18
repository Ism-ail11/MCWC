from __future__ import annotations

import argparse
import json
import os

import torch

from mcwc.codec.bitstream import read_bitstream
from mcwc.codec.mcwc_codec import MCWCCodec, MCWCConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="Output .pt path with decoded state_dict")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    header, _ = read_bitstream(args.bitstream)
    model_id = header.get("model_id", "")
    cfg_json = header.get("config_json", "")
    codec_cfg = header.get("codec", {})

    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore

    if cfg_json:
        cfg = AutoConfig.from_pretrained(model_id)  # fallback base
        # overwrite with serialized json if possible
        try:
            cfg = AutoConfig.from_dict(json.loads(cfg_json))
        except Exception:
            pass
    else:
        cfg = AutoConfig.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_config(cfg)

    codec = MCWCCodec(MCWCConfig(
        k=int(codec_cfg.get("k", 4)),
        qbits=int(codec_cfg.get("qbits", 8)),
        align=str(codec_cfg.get("align", "greedy")),
        compress=str(codec_cfg.get("compress", "zstd")),
        use_sortproj=bool(codec_cfg.get("use_sortproj", False)),
        sortproj_seed=int(codec_cfg.get("sortproj_seed", 0)),
    ))

    codec.decode(args.bitstream, model)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"[OK] Saved decoded state_dict: {args.out}")


if __name__ == "__main__":
    main()
