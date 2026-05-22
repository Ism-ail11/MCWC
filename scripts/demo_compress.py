from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path so this script works when run from scripts/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcwc.utils.compress import compress_bytes, decompress_bytes


def main() -> None:
    data = b"Hello MCWC demo!"
    compressed, method = compress_bytes(data, method="zstd")
    decompressed = decompress_bytes(compressed, method)

    print(f"method: {method}")
    print(f"original length: {len(data)}")
    print(f"compressed length: {len(compressed)}")
    print(f"roundtrip successful: {decompressed == data}")


if __name__ == "__main__":
    main()
