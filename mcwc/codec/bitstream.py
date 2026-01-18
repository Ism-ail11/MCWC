from __future__ import annotations

import json
import struct
from typing import Any, Dict, List, Tuple

import msgpack
import numpy as np

from mcwc.utils.compress import compress_bytes, decompress_bytes

MAGIC = b"MCWC"
VERSION = 0


def _pack_ndarray(arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    header = msgpack.packb(
        {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        },
        use_bin_type=True,
    )
    return struct.pack("<I", len(header)) + header + arr.tobytes(order="C")


def _unpack_ndarray(blob: bytes) -> np.ndarray:
    (hlen,) = struct.unpack("<I", blob[:4])
    header = msgpack.unpackb(blob[4:4 + hlen], raw=False)
    data = blob[4 + hlen :]
    arr = np.frombuffer(data, dtype=np.dtype(header["dtype"]))
    return arr.reshape(header["shape"])


def write_bitstream(path: str, header: Dict[str, Any], blobs: List[bytes], compress: str = "zstd") -> None:
    """Write bitstream to disk."""
    # compress blobs
    out_blobs: List[bytes] = []
    used_methods: List[str] = []
    for b in blobs:
        cb, method = compress_bytes(b, method=compress)
        out_blobs.append(cb)
        used_methods.append(method)

    header2 = dict(header)
    header2["magic"] = MAGIC.decode("ascii")
    header2["version"] = VERSION
    header2["blob_methods"] = used_methods

    h = msgpack.packb(header2, use_bin_type=True)

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(h)))
        f.write(h)
        f.write(struct.pack("<I", len(out_blobs)))
        for cb in out_blobs:
            f.write(struct.pack("<Q", len(cb)))
            f.write(cb)


def read_bitstream(path: str) -> Tuple[Dict[str, Any], List[bytes]]:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("Not an MCWC bitstream")
        (ver,) = struct.unpack("<I", f.read(4))
        (hlen,) = struct.unpack("<I", f.read(4))
        header = msgpack.unpackb(f.read(hlen), raw=False)
        (nblobs,) = struct.unpack("<I", f.read(4))
        blobs: List[bytes] = []
        for _ in range(nblobs):
            (blen,) = struct.unpack("<Q", f.read(8))
            blobs.append(f.read(blen))

    methods = header.get("blob_methods", ["zlib"] * len(blobs))
    out: List[bytes] = []
    for cb, method in zip(blobs, methods):
        out.append(decompress_bytes(cb, method))
    return header, out


# helpers to store arrays as blobs

def pack_array_blob(arr: np.ndarray) -> bytes:
    return _pack_ndarray(arr)


def unpack_array_blob(blob: bytes) -> np.ndarray:
    return _unpack_ndarray(blob)
