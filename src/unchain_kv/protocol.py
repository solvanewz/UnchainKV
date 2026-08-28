from __future__ import annotations

from dataclasses import dataclass
import json
import struct
from zlib import crc32


MAGIC = b"KVP0"
GRANT_MAGIC = b"KVG0"
HEADER_LEN_STRUCT = struct.Struct("!H")
HEADER_SIZE = len(MAGIC) + HEADER_LEN_STRUCT.size


@dataclass(frozen=True)
class ChunkHeader:
    transfer_id: str
    request_id: str
    layer_index: int
    block_index: int
    chunk_index: int
    chunks_in_layer: int
    offset: int
    payload_len: int
    checksum: int

    def to_bytes(self) -> bytes:
        return json.dumps(self.__dict__, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> ChunkHeader:
        values = json.loads(data.decode("utf-8"))
        return cls(**values)


@dataclass(frozen=True)
class Chunk:
    header: ChunkHeader
    payload: bytes

    def pack(self) -> bytes:
        header = self.header.to_bytes()
        if len(header) > 65535:
            raise ValueError("chunk header too large")
        if len(self.payload) != self.header.payload_len:
            raise ValueError("payload length does not match header")
        return MAGIC + HEADER_LEN_STRUCT.pack(len(header)) + header + self.payload

    @classmethod
    def unpack(cls, data: bytes) -> Chunk:
        if not data.startswith(MAGIC):
            raise ValueError("bad chunk magic")
        header_len = HEADER_LEN_STRUCT.unpack(data[len(MAGIC) : HEADER_SIZE])[0]
        header_start = HEADER_SIZE
        header_end = header_start + header_len
        header = ChunkHeader.from_bytes(data[header_start:header_end])
        payload = data[header_end:]
        if len(payload) != header.payload_len:
            raise ValueError("payload length mismatch")
        if (crc32(payload) & 0xFFFFFFFF) != header.checksum:
            raise ValueError("payload checksum mismatch")
        return cls(header, payload)


@dataclass(frozen=True)
class Grant:
    layer_index: int
    kind: str = "grant"
    transfer_id: str = ""

    def pack(self) -> bytes:
        header = json.dumps(self.__dict__, separators=(",", ":")).encode("utf-8")
        if len(header) > 65535:
            raise ValueError("grant header too large")
        return GRANT_MAGIC + HEADER_LEN_STRUCT.pack(len(header)) + header

    @classmethod
    def is_grant(cls, data: bytes) -> bool:
        return data.startswith(GRANT_MAGIC)

    @classmethod
    def unpack(cls, data: bytes) -> Grant:
        if not data.startswith(GRANT_MAGIC):
            raise ValueError("bad grant magic")
        header_len = HEADER_LEN_STRUCT.unpack(data[len(GRANT_MAGIC) : HEADER_SIZE])[0]
        header_start = HEADER_SIZE
        header_end = header_start + header_len
        values = json.loads(data[header_start:header_end].decode("utf-8"))
        return cls(**values)
