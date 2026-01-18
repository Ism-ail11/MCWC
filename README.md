# Motion-Compensated Weight Compression (MCWC)

This repository provides a **reference implementation** of **MCWC: Motion-Compensated Weight Compression**, a weight-only codec that treats network depth as a sequence and applies:

1. **Function-preserving alignment** (permutation symmetries) to remove "motion".
2. **Layer-sequential prediction** with **keyframes** and **residual coding**.
3. **Quantization + compressed bitstream** suitable for fast decode.

## Folder structure

```
motion-compensated-weight-compression/
  mcwc/                 # Python package
    align/              # assignment solvers + similarity
    codec/              # encoder/decoder, bitstream, quantizer, predictor
    data/               # dataset helpers (HF datasets / torchvision)
    eval/               # evaluation helpers (ppl/acc)
    models/             # HF model adapters (GPT2/OPT/NeoX/ViT)
    utils/              # misc helpers
  scripts/              # command-line entrypoints
  configs/              # example YAML configs
  tests/                # unit + smoke tests (toy offline)
  docs/                 # bitstream specification
```

## Setup

### Option A: Python venv (recommended)

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -e .
```

### Optional: HuggingFace + datasets

```bash
pip install -e ".[hf]"
```

### Optional: Zstandard compression

```bash
pip install -e ".[zstd]"
```

## Quick start (offline toy smoke test)

This test does **not** require transformers or internet.

```bash
python scripts/smoke_test_roundtrip.py
```

## HuggingFace end-to-end example (GPT-2)

```bash
python scripts/encode_model.py --model gpt2 --out artifacts/gpt2.mcwc --k 4 --qbits 8
python scripts/decode_model.py --bitstream artifacts/gpt2.mcwc --out artifacts/gpt2_decoded.pt
python scripts/evaluate_lm.py --model gpt2 --decoded artifacts/gpt2_decoded.pt --dataset wikitext-2 --split test
```

## Notes

- This is a **reference** codec: it prioritizes clarity and correctness.
- The bitstream format is documented in `docs/bitstream_spec.md`.
- Alignment currently targets **FFN hidden-unit permutations**. Attention-head alignment can be added following the same interface.

## License

MIT License (see `LICENSE`).
