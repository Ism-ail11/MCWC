from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from mcwc.align.similarity import cosine_similarity_matrix
from mcwc.align.assignment import AlignMethod, solve_alignment, solve_alignment_sortproj
from mcwc.codec.predictor import fit_affine, AffinePredictor
from mcwc.codec.quantizer import symmetric_quant_params, quantize, dequantize
from mcwc.codec.bitstream import write_bitstream, read_bitstream, pack_array_blob, unpack_array_blob
from mcwc.models.adapters import infer_ffn_specs_from_state_dict, FFNSpec


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _apply_perm_up(weight: torch.Tensor, perm: np.ndarray, family: str) -> torch.Tensor:
    """Apply permutation to FFN up-projection.

    For a standard Linear weight [out, in], permutes rows (out dimension).
    For GPT-2 Conv1D weight [in, out], permutes columns.
    """
    p = torch.from_numpy(perm.astype(np.int64))
    if family == "gpt2":
        return weight.index_select(1, p)
    else:
        return weight.index_select(0, p)


def _apply_perm_down(weight: torch.Tensor, perm: np.ndarray, family: str) -> torch.Tensor:
    """Apply permutation to FFN down-projection.

    For Linear down weight [out, in], permutes columns (in dimension).
    For GPT-2 Conv1D c_proj weight [in, out], permutes rows (in dimension).
    """
    p = torch.from_numpy(perm.astype(np.int64))
    if family == "gpt2":
        return weight.index_select(0, p)
    else:
        return weight.index_select(1, p)


def _extract_units_up(weight: torch.Tensor, family: str) -> np.ndarray:
    """Return matrix [H, d] of unit vectors for the up projection."""
    W = _to_np(weight)
    if family == "gpt2":
        # [in, H] -> units are columns
        return W.T
    else:
        # [H, in] -> units are rows
        return W


def _reassemble_units_up(units: np.ndarray, weight_shape: Tuple[int, ...], family: str) -> np.ndarray:
    if family == "gpt2":
        # units [H, in] -> weight [in, H]
        return units.T.reshape(weight_shape)
    else:
        return units.reshape(weight_shape)


@dataclass
class MCWCConfig:
    k: int = 4
    qbits: int = 8
    align: AlignMethod = "greedy"
    compress: str = "zstd"
    use_sortproj: bool = False
    sortproj_seed: int = 0


class MCWCCodec:
    """A reference MCWC codec focusing on FFN permutations + predictive residual coding."""

    def __init__(self, cfg: MCWCConfig):
        self.cfg = cfg

    def encode(self, model: torch.nn.Module, model_id: str, out_path: str, store_config_json: Optional[str] = None) -> None:
        state = model.state_dict()
        specs = infer_ffn_specs_from_state_dict(state)

        header: Dict[str, Any] = {
            "model_id": model_id,
            "config_json": store_config_json or "",
            "codec": {
                "k": int(self.cfg.k),
                "qbits": int(self.cfg.qbits),
                "align": str(self.cfg.align),
                "compress": str(self.cfg.compress),
                "use_sortproj": bool(self.cfg.use_sortproj),
                "sortproj_seed": int(self.cfg.sortproj_seed),
            },
            "tensors": [],
        }

        blobs: List[bytes] = []

        # We'll encode all tensors, but apply MCWC only to FFN up/down pairs.
        # For other tensors: store as keyframes with quantization.
        used = set()

        # Precompute layer order within each family separately.
        # We do MCWC per-family (gpt2 vs opt etc.)
        by_family: Dict[str, List[FFNSpec]] = {}
        for s in specs:
            by_family.setdefault(s.family, []).append(s)

        # Helper: encode a tensor as keyframe
        def encode_keyframe(name: str, t: torch.Tensor, group: str, layer: int) -> None:
            arr = _to_np(t)
            qp = symmetric_quant_params(arr, self.cfg.qbits)
            q = quantize(arr, qp)
            data_blob = len(blobs)
            blobs.append(pack_array_blob(q))
            header["tensors"].append({
                "name": name,
                "shape": list(arr.shape),
                "dtype": str(t.dtype),
                "role": "keyframe",
                "layer": int(layer),
                "group": group,
                "perm_blob": None,
                "data_blob": int(data_blob),
                "scale": float(qp.scale),
                "zero": float(qp.zero),
            })

        # Encode FFN pairs with alignment/prediction
        for family, fam_specs in by_family.items():
            fam_specs = sorted(fam_specs, key=lambda x: x.layer)
            prev_units: Optional[np.ndarray] = None
            prev_perm_units: Optional[np.ndarray] = None
            predictor: Optional[AffinePredictor] = None

            for idx, s in enumerate(fam_specs):
                W_up = state[s.up]
                W_dn = state[s.down]
                used.add(s.up); used.add(s.down)

                units = _extract_units_up(W_up, family)

                # choose permutation relative to previous layer (if available)
                perm = np.arange(units.shape[0], dtype=np.int32)
                if prev_units is not None:
                    if self.cfg.use_sortproj:
                        ar = solve_alignment_sortproj(prev_units, units, seed=self.cfg.sortproj_seed)
                    else:
                        sim = cosine_similarity_matrix(prev_units, units)
                        ar = solve_alignment(sim, method=self.cfg.align)
                    perm = ar.perm

                # apply alignment to current layer weights
                W_up_aligned = _apply_perm_up(W_up, perm, family)
                W_dn_aligned = _apply_perm_down(W_dn, perm, family)

                # extract aligned units for prediction fitting
                units_aligned = _extract_units_up(W_up_aligned, family)

                is_key = (s.layer % self.cfg.k) == 0

                if prev_perm_units is not None and (not is_key):
                    # fit predictor between previous aligned units and current aligned units
                    predictor = fit_affine(prev_perm_units, units_aligned)
                    pred = predictor.predict(prev_perm_units)
                    resid = (units_aligned - pred).astype(np.float32)

                    qp = symmetric_quant_params(resid, self.cfg.qbits)
                    q = quantize(resid, qp)
                    data_blob = len(blobs)
                    blobs.append(pack_array_blob(q))

                    # store permutation + residual blob
                    perm_blob = len(blobs)
                    blobs.append(pack_array_blob(perm.astype(np.int32)))

                    header["tensors"].append({
                        "name": s.up,
                        "shape": list(W_up.shape),
                        "dtype": str(W_up.dtype),
                        "role": "residual",
                        "layer": int(s.layer),
                        "group": "ffn_up",
                        "family": family,
                        "perm_blob": int(perm_blob),
                        "data_blob": int(data_blob),
                        "scale": float(qp.scale),
                        "zero": float(qp.zero),
                        "pred_a": float(predictor.a),
                        "pred_b": float(predictor.b),
                    })
                    # down tensor is stored as aligned keyframe at same layer (simpler reference codec)
                    encode_keyframe(s.down, W_dn_aligned, group="ffn_down", layer=s.layer)

                else:
                    # keyframe: store aligned weights directly
                    encode_keyframe(s.up, W_up_aligned, group="ffn_up", layer=s.layer)
                    encode_keyframe(s.down, W_dn_aligned, group="ffn_down", layer=s.layer)

                prev_units = units
                prev_perm_units = units_aligned

        # Encode remaining tensors as keyframes
        for name, t in state.items():
            if name in used:
                continue
            # store everything else directly
            encode_keyframe(name, t, group="other", layer=-1)

        write_bitstream(out_path, header=header, blobs=blobs, compress=self.cfg.compress)

    def decode(self, bitstream_path: str, model: torch.nn.Module) -> torch.nn.Module:
        header, blobs = read_bitstream(bitstream_path)
        records = header["tensors"]

        # load into state dict
        state = model.state_dict()

        # For residual-coded ffn_up, we need previous decoded aligned units
        prev_aligned_units_by_family: Dict[str, Optional[np.ndarray]] = {}

        for rec in records:
            name = rec["name"]
            role = rec["role"]
            group = rec.get("group", "other")
            family = rec.get("family", "")

            shape = tuple(rec["shape"])
            scale = float(rec["scale"])
            zero = float(rec.get("zero", 0.0))

            q = unpack_array_blob(blobs[rec["data_blob"]]).astype(np.int16)
            # dequant
            arr = (q.astype(np.float32) * scale + zero).reshape((-1,))

            if role == "keyframe":
                arr = arr.reshape(shape).astype(np.float32)
                t = torch.from_numpy(arr)
                # cast to original dtype
                t = t.to(dtype=state[name].dtype)
                state[name].copy_(t)

                # update prev aligned units for ffn_up keyframes
                if group == "ffn_up" and family:
                    prev_aligned_units_by_family[family] = _extract_units_up(state[name], family)

            elif role == "residual":
                if not family:
                    raise ValueError("Residual record missing family")

                resid_units = arr.reshape((-1,))
                # resid is [H, d]
                H = int(rec["shape"][1] if family == "gpt2" else rec["shape"][0])
                # infer d from residual length
                d = int(resid_units.size // H)
                resid = resid_units.reshape((H, d)).astype(np.float32)

                prev_units = prev_aligned_units_by_family.get(family)
                if prev_units is None:
                    raise ValueError(f"Missing previous aligned units for family={family}")

                a = float(rec["pred_a"])
                b = float(rec["pred_b"])
                pred = (a * prev_units + b).astype(np.float32)
                cur_aligned = (pred + resid).astype(np.float32)

                # permutation blob for alignment
                perm = unpack_array_blob(blobs[rec["perm_blob"]]).astype(np.int32)

                # cur_aligned corresponds to aligned up units. Need to place into weight with aligned order,
                # but we stored perm mapping from prev->cur before alignment. In encoding we applied perm to align.
                # For decode we reconstruct aligned tensor directly and store it.
                W_np = _reassemble_units_up(cur_aligned, tuple(rec["shape"]), family)
                t = torch.from_numpy(W_np).to(dtype=state[name].dtype)
                state[name].copy_(t)

                prev_aligned_units_by_family[family] = _extract_units_up(state[name], family)

            else:
                raise ValueError(f"Unknown role: {role}")

        model.load_state_dict(state)
        return model
