# MCWC Bitstream Specification (v0)

An MCWC bitstream is a single binary file containing:

1. **Header** (MessagePack map)
2. **Payload blobs** (compressed byte arrays)

## Header fields

- `version` (int): bitstream format version.
- `model_id` (str): identifier (e.g., `gpt2`, `facebook/opt-1.3b`).
- `config_json` (str): optional serialized model config (JSON) for standalone decode.
- `codec` (map): codec hyperparameters
  - `k` (int): keyframe interval
  - `qbits` (int): quantization bits
  - `align` (str): `greedy` | `hungarian` | `sortproj`
  - `compress` (str): `zstd` | `zlib`
- `tensors` (list[map]): list of tensor records
  - `name` (str): state_dict key
  - `shape` (list[int])
  - `dtype` (str)
  - `role` (str): `keyframe` | `residual`
  - `layer` (int): layer index (if applicable)
  - `group` (str): grouping label (`ffn_up`, `ffn_down`, or `other`)
  - `perm_blob` (int|null): payload index for permutation array
  - `data_blob` (int): payload index for quantized codes
  - `scale` (float): quantization scale

## Payload blobs

A list of byte strings. Each blob is compressed (`zstd` if available else `zlib`).

- Permutation blobs store an `int32` array (little-endian) of length `H`.
- Data blobs store an `int16` array of quantized symbols.

See `mcwc/codec/bitstream.py` for exact packing/unpacking.
