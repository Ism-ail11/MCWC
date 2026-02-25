import math
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np  # type: ignore[import-untyped]
import torch  # type: ignore[import-untyped]
import msgpack  # type: ignore[import-untyped]
import zstandard as zstd  # type: ignore[import-untyped]
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

import re
_LAY_PATTERNS = [
    re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)h\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)"),
]

def infer_layer_index(param_name: str) -> Optional[int]:
    for pat in _LAY_PATTERNS:
        m = pat.search(param_name)
        if m:
            return int(m.group(1))
    return None

def is_ffn_fc1(name: str) -> bool:
    return any(s in name for s in [
        ".mlp.fc1.weight", ".mlp.c_fc.weight", ".mlp.dense_h_to_4h.weight",
        ".fc1.weight", ".c_fc.weight", "dense_h_to_4h.weight"
    ])

def paired_fc2_name(fc1_name: str) -> Optional[str]:
    repl = [
        (".mlp.fc1.weight", ".mlp.fc2.weight"),
        (".mlp.c_fc.weight", ".mlp.c_proj.weight"),
        (".mlp.dense_h_to_4h.weight", ".mlp.dense_4h_to_h.weight"),
        (".fc1.weight", ".fc2.weight"),
        (".c_fc.weight", ".c_proj.weight"),
        ("dense_h_to_4h.weight", "dense_4h_to_h.weight"),
    ]
    for a,b in repl:
        if a in fc1_name:
            return fc1_name.replace(a,b)
    return None

@dataclass
class FFNPair:
    layer: int
    fc1: str
    fc2: str

def find_ffn_pairs(state_dict_keys):
    keys = set(state_dict_keys)
    pairs = []
    for k in state_dict_keys:
        if not is_ffn_fc1(k):
            continue
        layer = infer_layer_index(k)
        if layer is None:
            continue
        fc2 = paired_fc2_name(k)
        if fc2 and fc2 in keys:
            pairs.append(FFNPair(layer=layer, fc1=k, fc2=fc2))
    pairs.sort(key=lambda p: p.layer)
    return pairs

def permute_rows(x, perm): return x[perm, :]
def permute_cols(x, perm): return x[:, perm]
def inverse_perm(perm):
    inv = [0]*len(perm)
    for i,p in enumerate(perm):
        inv[p] = i
    return inv

@dataclass
class CodecConfig:
    keyframe_interval: int = 4
    qmax: int = 2047
    use_hungarian_if_leq: int = 512
    predictor_ridge: float = 1e-4
    zstd_level: int = 9
    device: str = "cpu"

@dataclass
class QuantParams:
    scale: torch.Tensor
    mean: torch.Tensor

class ScalarQuantizer:
    def __init__(self, qmax: int):
        self.qmax = int(qmax)

    def fit(self, x: torch.Tensor, per_channel_dim: Optional[int] = None) -> QuantParams:
        if per_channel_dim is None:
            mean = x.mean()
            std = x.std().clamp_min(1e-8)
            scale = (3.0 * std) / float(self.qmax)
            return QuantParams(scale=scale, mean=mean)
        x2 = x.movedim(per_channel_dim, 0).contiguous()
        C = x2.shape[0]
        flat = x2.reshape(C, -1)
        mean = flat.mean(dim=1)
        std = flat.std(dim=1).clamp_min(1e-8)
        scale = (3.0 * std) / float(self.qmax)
        return QuantParams(scale=scale, mean=mean)

    def quantize(self, x: torch.Tensor, qp: QuantParams, per_channel_dim: Optional[int] = None) -> torch.Tensor:
        if per_channel_dim is None:
            y = torch.round((x - qp.mean) / qp.scale).clamp(-self.qmax, self.qmax)
            return y.to(torch.int16)
        x2 = x.movedim(per_channel_dim, 0).contiguous()
        C = x2.shape[0]
        flat = x2.reshape(C, -1)
        q = torch.round((flat - qp.mean[:, None]) / qp.scale[:, None]).clamp(-self.qmax, self.qmax)
        q = q.to(torch.int16).reshape_as(x2)
        return q.movedim(0, per_channel_dim).contiguous()

    def dequantize(self, q: torch.Tensor, qp: QuantParams, per_channel_dim: Optional[int] = None) -> torch.Tensor:
        if per_channel_dim is None:
            return q.to(torch.float32) * qp.scale + qp.mean
        q2 = q.movedim(per_channel_dim, 0).contiguous()
        C = q2.shape[0]
        flat = q2.reshape(C, -1).to(torch.float32)
        x = flat * qp.scale[:, None] + qp.mean[:, None]
        x = x.reshape_as(q2)
        return x.movedim(0, per_channel_dim).contiguous()

class AffinePredictor:
    def __init__(self, ridge: float = 1e-4):
        self.ridge = float(ridge)
        self.a: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        meanx = X.mean(dim=1)
        meany = Y.mean(dim=1)
        cov = ((X - meanx[:,None]) * (Y - meany[:,None])).sum(dim=1)
        var = ((X - meanx[:,None]) * (X - meanx[:,None])).sum(dim=1) + self.ridge
        self.a = (cov / var).detach()
        self.b = (meany - self.a * meanx).detach()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.a is None or self.b is None:
            return X
        a: torch.Tensor = self.a
        b: torch.Tensor = self.b
        return a[:, None] * X + b[:, None]  # type: ignore[index]

def _channel_representation(fc1_w: torch.Tensor, fc2_w: torch.Tensor) -> torch.Tensor:
    rep = torch.cat([fc1_w, fc2_w.t()], dim=1)
    rep = rep / (rep.norm(dim=1, keepdim=True).clamp_min(1e-8))
    return rep

def solve_permutation(rep_prev: torch.Tensor, rep_cur: torch.Tensor, use_hungarian_if_leq: int = 512) -> List[int]:
    H = rep_prev.shape[0]
    S = (rep_prev @ rep_cur.t()).cpu().numpy()
    if H <= use_hungarian_if_leq:
        row, col = linear_sum_assignment(-S)
        perm = [0]*H
        for r,c in zip(row, col):
            perm[int(r)] = int(c)
        return perm
    remaining = set(range(H))
    perm = [-1]*H
    for i in range(H):
        j = int(np.argmax(S[i]))
        if j in remaining:
            perm[i] = j
            remaining.remove(j)
        else:
            if not remaining:
                raise RuntimeError(f"Greedy permutation solver exhausted candidates at index {i}")
            bestj, bestv = next(iter(remaining)), -1e9
            for cand in remaining:
                v = S[i, cand]  # type: ignore[index]
                if v > bestv:
                    bestv, bestj = v, cand
            perm[i] = int(bestj)
            remaining.remove(int(bestj))
    return perm

MAX_BLOB_BYTES = 10 * 1024 * 1024 * 1024   # 10 GB safety cap

def _pack_int16(q: torch.Tensor) -> bytes:
    return q.detach().cpu().numpy().astype(np.int16).tobytes()

def _unpack_int16(buf: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    expected = int(np.prod(shape)) * 2   # int16 = 2 bytes
    if len(buf) != expected:
        raise ValueError(f"int16 buffer size {len(buf)} != expected {expected} for shape {shape}")
    arr = np.frombuffer(buf, dtype=np.int16).reshape(shape)
    return torch.from_numpy(arr.copy())

def _pack_fp16(x: torch.Tensor) -> bytes:
    return x.detach().cpu().numpy().astype(np.float16).tobytes()

def _unpack_fp16(buf: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    expected = int(np.prod(shape)) * 2   # fp16 = 2 bytes
    if len(buf) != expected:
        raise ValueError(f"fp16 buffer size {len(buf)} != expected {expected} for shape {shape}")
    arr = np.frombuffer(buf, dtype=np.float16).reshape(shape)
    return torch.from_numpy(arr.copy()).to(torch.float32)

class Bitstream:
    def __init__(self, zstd_level: int = 9):
        self.cctx = zstd.ZstdCompressor(level=int(zstd_level))
        self.dctx = zstd.ZstdDecompressor()

    def dumps(self, obj: Any) -> bytes:
        raw = msgpack.packb(obj, use_bin_type=True)
        compressed = self.cctx.compress(raw)
        digest = hashlib.sha256(compressed).digest()   # 32 bytes
        return digest + compressed

    def loads(self, data: bytes) -> Dict[str, Any]:
        if len(data) < 32:
            raise ValueError("Bitstream too short — missing integrity digest")
        stored_digest: bytes = data[:32]  # type: ignore[assignment]
        compressed: bytes = data[32:]  # type: ignore[assignment]
        if hashlib.sha256(compressed).digest() != stored_digest:
            raise ValueError("Integrity check failed — bitstream is corrupted or tampered")
        raw = self.dctx.decompress(compressed)
        result: Dict[str, Any] = msgpack.unpackb(raw, raw=False)
        return result

class MCWCCodec:
    def __init__(self, cfg: CodecConfig):
        self.cfg = cfg
        self.bs = Bitstream(zstd_level=cfg.zstd_level)
        self.quant = ScalarQuantizer(qmax=cfg.qmax)

    def encode_state_dict(self, state: Dict[str, torch.Tensor]) -> bytes:
        device = torch.device(self.cfg.device)
        state = {k: v.detach().to(device=device, dtype=torch.float32) for k,v in state.items()}

        ffn_pairs = find_ffn_pairs(state.keys())
        layers = sorted({p.layer for p in ffn_pairs})
        pair_by_layer = {p.layer: p for p in ffn_pairs}

        header: Dict[str, Any] = {
            "format": "mcwc_onefolder_ref_v1",
            "config": asdict(self.cfg),  # type: ignore[arg-type]
            "tensors": {k: {"shape": list(v.shape)} for k,v in state.items()},
            "ffn_layers": layers,
            "ffn_pairs": [{"layer": p.layer, "fc1": p.fc1, "fc2": p.fc2} for p in ffn_pairs],
        }

        records = []
        prev_rep = None
        predictor = AffinePredictor(ridge=self.cfg.predictor_ridge)

        for li, layer in enumerate(layers):
            p = pair_by_layer[layer]
            fc1 = state[p.fc1]
            fc2 = state[p.fc2]
            H, D = fc1.shape
            rep = _channel_representation(fc1, fc2)

            if prev_rep is None:
                perm = list(range(H))
            else:
                perm = solve_permutation(prev_rep, rep, use_hungarian_if_leq=self.cfg.use_hungarian_if_leq)

            fc1_al = permute_rows(fc1, perm)
            fc2_al = permute_cols(fc2, perm)
            rep_al = _channel_representation(fc1_al, fc2_al)

            is_key = (li % self.cfg.keyframe_interval == 0) or (prev_rep is None)

            if is_key:
                qp1 = self.quant.fit(fc1_al, per_channel_dim=0)
                q1  = self.quant.quantize(fc1_al, qp1, per_channel_dim=0)
                qp2 = self.quant.fit(fc2_al, per_channel_dim=1)
                q2  = self.quant.quantize(fc2_al, qp2, per_channel_dim=1)

                rec = {
                    "layer": int(layer),
                    "kind": "keyframe",
                    "perm": perm,
                    "fc1_q_shape": list(q1.shape),
                    "fc2_q_shape": list(q2.shape),
                    "fc1_q": _pack_int16(q1),
                    "fc2_q": _pack_int16(q2),
                    "s1_shape": list(qp1.scale.shape),
                    "m1_shape": list(qp1.mean.shape),
                    "s2_shape": list(qp2.scale.shape),
                    "m2_shape": list(qp2.mean.shape),
                    "s1": _pack_fp16(qp1.scale),
                    "m1": _pack_fp16(qp1.mean),
                    "s2": _pack_fp16(qp2.scale),
                    "m2": _pack_fp16(qp2.mean),
                }
            else:
                # Store per-row norms so residual decode can restore magnitudes
                row_norms = rep_al.norm(dim=1, keepdim=False).clamp_min(1e-8)

                predictor.fit(prev_rep, rep_al)
                rep_hat = predictor.predict(prev_rep)
                resid = rep_al - rep_hat

                qp = self.quant.fit(resid, per_channel_dim=0)
                q  = self.quant.quantize(resid, qp, per_channel_dim=0)

                rec = {
                    "layer": int(layer),
                    "kind": "residual",
                    "perm": perm,
                    "q_shape": list(q.shape),
                    "q": _pack_int16(q),
                    "s_shape": list(qp.scale.shape),
                    "m_shape": list(qp.mean.shape),
                    "s": _pack_fp16(qp.scale),
                    "m": _pack_fp16(qp.mean),
                }
                pred_a = predictor.a
                pred_b = predictor.b
                if pred_a is not None and pred_b is not None:
                    rec["a_shape"] = [int(pred_a.numel())]
                    rec["b_shape"] = [int(pred_b.numel())]
                    rec["a"] = _pack_fp16(pred_a)
                    rec["b"] = _pack_fp16(pred_b)
                else:
                    rec["a_shape"] = [0]
                    rec["b_shape"] = [0]
                    rec["a"] = b""
                    rec["b"] = b""
                rec["norms_shape"] = [int(row_norms.numel())]
                rec["norms"] = _pack_fp16(row_norms)

            records.append(rec)
            prev_rep = rep_al.detach()

        ffn_keys = set()
        for p in ffn_pairs:
            ffn_keys.add(p.fc1); ffn_keys.add(p.fc2)

        other = []
        for k,v in state.items():
            if k in ffn_keys:
                continue
            qp = self.quant.fit(v, per_channel_dim=None)
            q  = self.quant.quantize(v, qp, per_channel_dim=None)
            other.append({
                "name": k,
                "shape": list(q.shape),
                "q": _pack_int16(q),
                "s": _pack_fp16(qp.scale.reshape(-1)),
                "m": _pack_fp16(qp.mean.reshape(-1)),
            })

        return self.bs.dumps({"header": header, "records": records, "other": other})

    def decode_state_dict(self, blob: bytes, template_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = torch.device(self.cfg.device)
        payload: Dict[str, Any] = self.bs.loads(blob)
        header: Dict[str, Any] = payload["header"]
        records: List[Dict[str, Any]] = payload["records"]
        other: List[Dict[str, Any]] = payload.get("other", [])

        out: Dict[str, torch.Tensor] = {k: template_state[k].detach().to(device=device, dtype=torch.float32) for k in template_state.keys()}

        for rec in other:
            name: str = str(rec["name"])
            shape = tuple(int(x) for x in rec["shape"])
            q = _unpack_int16(bytes(rec["q"]), shape).to(device)
            s = _unpack_fp16(bytes(rec["s"]), (1,)).to(device)[0]
            m = _unpack_fp16(bytes(rec["m"]), (1,)).to(device)[0]
            x = self.quant.dequantize(q, QuantParams(scale=s, mean=m), per_channel_dim=None)
            out[name] = x

        prev_rep: Optional[torch.Tensor] = None
        ffn_pairs_list: List[Dict[str, Any]] = header["ffn_pairs"]
        pair_by_layer_dec: Dict[int, Dict[str, Any]] = {
            int(p["layer"]): p for p in ffn_pairs_list
        }
        for rec in records:
            layer: int = int(rec["layer"])
            pair = pair_by_layer_dec.get(layer)
            if pair is None:
                continue
            assert pair is not None
            fc1_name: str = str(pair["fc1"])
            fc2_name: str = str(pair["fc2"])

            perm: List[int] = list(rec["perm"])
            inv = inverse_perm(perm)

            if rec["kind"] == "keyframe":
                fc1_q_shape = tuple(int(x) for x in rec["fc1_q_shape"])
                fc2_q_shape = tuple(int(x) for x in rec["fc2_q_shape"])
                s1_shape = tuple(int(x) for x in rec["s1_shape"])
                m1_shape = tuple(int(x) for x in rec["m1_shape"])
                s2_shape = tuple(int(x) for x in rec["s2_shape"])
                m2_shape = tuple(int(x) for x in rec["m2_shape"])
                q1 = _unpack_int16(bytes(rec["fc1_q"]), fc1_q_shape).to(device)
                q2 = _unpack_int16(bytes(rec["fc2_q"]), fc2_q_shape).to(device)
                s1 = _unpack_fp16(bytes(rec["s1"]), s1_shape).to(device)
                m1 = _unpack_fp16(bytes(rec["m1"]), m1_shape).to(device)
                s2 = _unpack_fp16(bytes(rec["s2"]), s2_shape).to(device)
                m2 = _unpack_fp16(bytes(rec["m2"]), m2_shape).to(device)
                fc1_al = self.quant.dequantize(q1, QuantParams(scale=s1, mean=m1), per_channel_dim=0)
                fc2_al = self.quant.dequantize(q2, QuantParams(scale=s2, mean=m2), per_channel_dim=1)
                rep_al = _channel_representation(fc1_al, fc2_al)
            else:
                q_shape = tuple(int(x) for x in rec["q_shape"])
                s_shape = tuple(int(x) for x in rec["s_shape"])
                m_shape = tuple(int(x) for x in rec["m_shape"])
                q = _unpack_int16(bytes(rec["q"]), q_shape).to(device)
                s = _unpack_fp16(bytes(rec["s"]), s_shape).to(device)
                m = _unpack_fp16(bytes(rec["m"]), m_shape).to(device)
                resid = self.quant.dequantize(q, QuantParams(scale=s, mean=m), per_channel_dim=0)

                a_shape = tuple(int(x) for x in rec["a_shape"])
                b_shape = tuple(int(x) for x in rec["b_shape"])
                norms_shape = tuple(int(x) for x in rec["norms_shape"])
                a = _unpack_fp16(bytes(rec["a"]), a_shape).to(device).reshape(-1)
                b = _unpack_fp16(bytes(rec["b"]), b_shape).to(device).reshape(-1)
                rep_hat = a[:, None] * prev_rep + b[:, None]
                rep_al = rep_hat + resid

                # Restore original weight magnitudes from saved norms
                norms = _unpack_fp16(bytes(rec["norms"]), norms_shape).to(device).reshape(-1)
                cur_norms = rep_al.norm(dim=1, keepdim=False).clamp_min(1e-8)
                rep_al = rep_al * (norms / cur_norms).unsqueeze(1)

                D = rep_al.shape[1] // 2
                fc1_al = rep_al[:, :D]
                fc2_al = rep_al[:, D:].t().contiguous()

            fc1 = permute_rows(fc1_al, inv)
            fc2 = permute_cols(fc2_al, inv)
            out[fc1_name] = fc1
            out[fc2_name] = fc2
            prev_rep = rep_al.detach()

        return out
