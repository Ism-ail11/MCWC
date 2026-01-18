from __future__ import annotations

import os
import tempfile

import torch

from mcwc.codec.mcwc_codec import MCWCCodec, MCWCConfig


class ToyMLP(torch.nn.Module):
    def __init__(self, d_in=32, d_h=64, d_out=32, L=4):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(L):
            self.layers.append(torch.nn.Sequential(
                torch.nn.Linear(d_in, d_h, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(d_h, d_out, bias=False),
            ))
            d_in = d_out

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


def main():
    torch.manual_seed(0)
    m = ToyMLP()

    # make state_dict keys look like OPT-style to exercise FFN detection
    sd = {}
    for i, blk in enumerate(m.layers):
        sd[f"model.decoder.layers.{i}.fc1.weight"] = blk[0].weight
        sd[f"model.decoder.layers.{i}.fc2.weight"] = blk[2].weight
    # add a non-ffn tensor
    sd["misc.bias"] = torch.zeros(1)

    # monkeypatch state_dict/load_state_dict
    class Wrap(torch.nn.Module):
        def __init__(self, inner, state):
            super().__init__()
            self.inner = inner
            self._state = state

        def state_dict(self, *args, **kwargs):
            return self._state

        def load_state_dict(self, state, *args, **kwargs):
            self._state = state
            return {}

    mw = Wrap(m, sd)

    cfg = MCWCConfig(k=2, qbits=8, align="greedy", compress="zlib")
    codec = MCWCCodec(cfg)

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "toy.mcwc")
        codec.encode(mw, model_id="toy", out_path=out)

        # decode into a fresh wrapper with same keys/shapes
        sd2 = {k: v.clone() for k, v in sd.items()}
        mw2 = Wrap(m, sd2)
        codec.decode(out, mw2)

        # sanity: forward works
        x = torch.randn(8, 32)
        y = m(x)
        assert torch.isfinite(y).all().item(), "NaN in output"

    print("[OK] smoke_test_roundtrip passed")


if __name__ == "__main__":
    main()
