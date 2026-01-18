from __future__ import annotations

import zlib
from typing import Tuple


def compress_bytes(data: bytes, method: str = "zstd", level: int = 3) -> Tuple[bytes, str]:
    """Compress bytes.

    If `method=='zstd'` but zstandard is not installed, falls back to zlib.
    Returns (compressed_bytes, used_method).
    """
    if method == "zstd":
        try:
            import zstandard as zstd  # type: ignore

            cctx = zstd.ZstdCompressor(level=level)
            return cctx.compress(data), "zstd"
        except Exception:
            pass

    # zlib fallback (always available)
    return zlib.compress(data, level=max(1, min(level, 9))), "zlib"


def decompress_bytes(data: bytes, method: str) -> bytes:
    """Decompress bytes for the given method."""
    if method == "zstd":
        import zstandard as zstd  # type: ignore

        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)

    if method == "zlib":
        return zlib.decompress(data)

    raise ValueError(f"Unknown compression method: {method}")
