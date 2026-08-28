from __future__ import annotations

from collections.abc import Iterable, Iterator


def split_block_bytes(data: bytes, chunk_size: int) -> Iterator[tuple[int, bytes]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for offset in range(0, len(data), chunk_size):
        yield offset, data[offset : offset + chunk_size]


def merge_block_bytes(size: int, chunks: Iterable[tuple[int, bytes]]) -> bytes:
    out = bytearray(size)
    for offset, payload in chunks:
        end = offset + len(payload)
        if offset < 0 or end > size:
            raise ValueError("chunk outside block")
        out[offset:end] = payload
    return bytes(out)
