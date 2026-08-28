from __future__ import annotations

from collections.abc import Iterable
import os
import socket
import struct
import threading
import time
from zlib import crc32

from .grant_state import GrantStore
from .layer_state import LayerStore
from .protocol import Chunk, ChunkHeader, GRANT_MAGIC, Grant


_FRAME_HEADER = struct.Struct("!II")
_U32 = struct.Struct("!I")
_BLOCK_LAYER_MAGIC = b"KVB1"
_KV_MAJOR_LAYER_MAGIC = b"KVK1"
_NATIVE_LAYER_MAGIC = b"KVN1"
_COMPRESSED_NATIVE_LAYER_MAGIC = b"KVC1"
_COMPRESSED_NATIVE_CHUNK_MAGIC = b"KVC2"
_GROUP_LAYER_MAGIC = b"KVG1"
_LAYOUT_CODES = {"block_major": 0, "native": 1}
_CODE_LAYOUTS = {value: key for key, value in _LAYOUT_CODES.items()}
_CODEC_CODES = {"raw_passthrough": 0, "splitzip_bf16": 1}
_CODE_CODECS = {value: key for key, value in _CODEC_CODES.items()}


class TcpReceiver:
    def __init__(
        self,
        bind: tuple[str, int],
        store: LayerStore,
        trace=None,
        max_bytes: int = 128 * 1024 * 1024,
        recv_buffer_bytes: int | None = None,
        grants: GrantStore | None = None,
    ) -> None:
        self.store = store
        self.grants = grants
        self.trace = trace
        self.max_bytes = max_bytes
        self._closed = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if recv_buffer_bytes is None:
            recv_buffer_bytes = int(
                os.environ.get("UNCHAIN_KV_RECV_BUFFER_BYTES", 16 * 1024 * 1024)
            )
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buffer_bytes)
        self._sock.bind(bind)
        self._sock.listen()
        self._sock.settimeout(0.1)
        self.port = self._sock.getsockname()[1]

    def serve(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with conn:
                try:
                    self._serve_conn(conn)
                except Exception as exc:
                    if self.trace is not None:
                        self.trace.event(
                            "tcp_receive_error",
                            error=type(exc).__name__,
                            detail=str(exc)[:160],
                        )
                    self.store.fail(exc)

    def close(self) -> None:
        self._closed.set()
        self._sock.close()

    def _serve_conn(self, conn: socket.socket) -> None:
        while not self._closed.is_set():
            header = _recv_exact(conn, _FRAME_HEADER.size, clean_eof=True)
            if header is None:
                return
            size, checksum = _FRAME_HEADER.unpack(header)
            if size > self.max_bytes:
                raise ValueError(f"tcp frame too large: {size}")
            data = _recv_exact(conn, size)
            _validate_frame(data, checksum)
            self._handle_frame(data)

    def _handle_frame(self, data: object) -> None:
        if _starts_with(data, GRANT_MAGIC):
            grant = Grant.unpack(bytes(data))
            if self.grants is not None:
                self.grants.add(
                    grant.layer_index,
                    kind=grant.kind,
                    transfer_id=grant.transfer_id,
                )
            if self.trace is not None:
                self.trace.event("grant_recv", layer=grant.layer_index)
            return
        if _starts_with(data, _BLOCK_LAYER_MAGIC):
            self._handle_block_layer_payload(data, "block_major")
            return
        if _starts_with(data, _NATIVE_LAYER_MAGIC):
            self._handle_block_layer_payload(data, "native")
            return
        if _starts_with(data, _KV_MAJOR_LAYER_MAGIC):
            self._handle_kv_major_layer_payload(data)
            return
        if _starts_with(data, _COMPRESSED_NATIVE_LAYER_MAGIC):
            self._handle_compressed_native_layer_payload(data)
            return
        if _starts_with(data, _COMPRESSED_NATIVE_CHUNK_MAGIC):
            self._handle_compressed_native_layer_payload(data, chunked=True)
            return
        if _starts_with(data, _GROUP_LAYER_MAGIC):
            self._handle_layer_group_payload(data)
            return
        chunk = Chunk.unpack(bytes(data))
        if self.trace is not None:
            self.trace.event(
                "chunk_recv",
                layer=chunk.header.layer_index,
                chunk=chunk.header.chunk_index,
                chunks=chunk.header.chunks_in_layer,
            )
        ready = self.store.add(chunk)
        if ready and self.trace is not None:
            self.trace.event(
                "layer_ready",
                layer=chunk.header.layer_index,
                chunks=chunk.header.chunks_in_layer,
            )

    def _handle_block_layer_payload(self, data: bytes, layout: str) -> None:
        parse_start = time.perf_counter()
        offset = len(_NATIVE_LAYER_MAGIC)
        transfer_id, offset = _read_string(data, offset)
        request_id, offset = _read_string(data, offset)
        layer_index = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        count = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        block_size = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        if len(data) - offset != count * block_size:
            raise ValueError(f"bad {layout} layer payload size")
        payload = memoryview(data)[offset:]
        ready = self.store.add(
            Chunk(
                ChunkHeader(
                    transfer_id,
                    request_id,
                    layer_index,
                    0,
                    0,
                    1,
                    0,
                    len(payload),
                    0,
                ),
                payload,
            ),
            layout=layout,
        )
        if ready and self.trace is not None:
            self.trace.event(
                "layer_ready",
                layer=layer_index,
                chunks=count,
                bytes=len(data),
                layout=layout,
                recv_copy_elapsed_s=0.0,
                recv_parse_store_elapsed_s=time.perf_counter() - parse_start,
            )

    def _handle_kv_major_layer_payload(self, data: bytes) -> None:
        parse_start = time.perf_counter()
        offset = len(_KV_MAJOR_LAYER_MAGIC)
        transfer_id, offset = _read_string(data, offset)
        request_id, offset = _read_string(data, offset)
        layer_index = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        count = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        part_size = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        part_bytes = count * part_size
        if len(data) - offset != part_bytes * 2:
            raise ValueError("bad kv-major layer payload size")
        ready = False
        for index, payload in enumerate(
            (
                memoryview(data)[offset : offset + part_bytes],
                memoryview(data)[offset + part_bytes :],
            )
        ):
            ready = self.store.add(
                Chunk(
                    ChunkHeader(
                        transfer_id,
                        request_id,
                        layer_index,
                        index,
                        index,
                        2,
                        0,
                        len(payload),
                        0,
                    ),
                    payload,
                ),
                layout="kv_major",
            )
        if ready and self.trace is not None:
            self.trace.event(
                "layer_ready",
                layer=layer_index,
                chunks=count,
                bytes=len(data),
                layout="kv_major",
                recv_copy_elapsed_s=0.0,
                recv_parse_store_elapsed_s=time.perf_counter() - parse_start,
            )

    def _handle_layer_group_payload(self, data: bytes) -> None:
        parse_start = time.perf_counter()
        offset = len(_GROUP_LAYER_MAGIC)
        transfer_id, offset = _read_string(data, offset)
        request_id, offset = _read_string(data, offset)
        first_layer = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        layer_count = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        block_count = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        block_size = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        layout_code = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        layout = _CODE_LAYOUTS.get(layout_code)
        if layout is None:
            raise ValueError("unknown group layer layout")
        layer_bytes = block_count * block_size
        if len(data) - offset != layer_count * layer_bytes:
            raise ValueError("bad layer group payload size")
        for index in range(layer_count):
            start = offset + index * layer_bytes
            payload = memoryview(data)[start : start + layer_bytes]
            ready = self.store.add(
                Chunk(
                    ChunkHeader(
                        transfer_id,
                        request_id,
                        first_layer + index,
                        0,
                        0,
                        1,
                        0,
                        len(payload),
                        0,
                    ),
                    payload,
                ),
                layout=layout,
            )
            if ready and self.trace is not None:
                self.trace.event(
                    "layer_ready",
                    layer=first_layer + index,
                    chunks=block_count,
                    bytes=len(payload),
                    layout=layout,
                    recv_copy_elapsed_s=0.0,
                    recv_parse_store_elapsed_s=time.perf_counter() - parse_start,
                )

    def _handle_compressed_native_layer_payload(
        self, data: bytes, chunked: bool = False
    ) -> None:
        parse_start = time.perf_counter()
        offset = len(
            _COMPRESSED_NATIVE_CHUNK_MAGIC
            if chunked
            else _COMPRESSED_NATIVE_LAYER_MAGIC
        )
        transfer_id, offset = _read_string(data, offset)
        request_id, offset = _read_string(data, offset)
        layer_index = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        block_count = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        raw_block_size = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        raw_bytes = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        codec_code = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        if chunked:
            chunk_index = _U32.unpack(data[offset : offset + _U32.size])[0]
            offset += _U32.size
            chunks_in_layer = _U32.unpack(data[offset : offset + _U32.size])[0]
            offset += _U32.size
        else:
            chunk_index = 0
            chunks_in_layer = 1
        encoded_bytes = _U32.unpack(data[offset : offset + _U32.size])[0]
        offset += _U32.size
        codec = _CODE_CODECS.get(codec_code)
        if codec is None:
            raise ValueError("unknown compressed native codec")
        if len(data) - offset != encoded_bytes:
            raise ValueError("bad compressed native payload size")
        payload = memoryview(data)[offset:]
        metadata = {
            "block_count": block_count,
            "raw_block_size": raw_block_size,
            "raw_bytes": raw_bytes,
            "codec": codec,
        }
        if chunked:
            metadata["chunks_in_layer"] = chunks_in_layer
        else:
            metadata["encoded_bytes"] = encoded_bytes
        ready = self.store.add(
            Chunk(
                ChunkHeader(
                    transfer_id,
                    request_id,
                    layer_index,
                    0,
                    chunk_index,
                    chunks_in_layer,
                    0,
                    len(payload),
                    0,
                ),
                payload,
            ),
            layout="compressed_native",
            metadata=metadata,
        )
        if ready and self.trace is not None:
            self.trace.event(
                "layer_ready",
                layer=layer_index,
                chunks=block_count,
                bytes=len(data),
                layout="compressed_native",
                codec=codec,
                raw_bytes=raw_bytes,
                encoded_bytes=encoded_bytes,
                chunk=chunk_index,
                frame_chunks=chunks_in_layer,
                recv_copy_elapsed_s=0.0,
                recv_parse_store_elapsed_s=time.perf_counter() - parse_start,
            )


def send_chunks(peer: tuple[str, int], chunks: Iterable[Chunk]) -> None:
    with _connect(peer) as sock:
        for chunk in chunks:
            _send_frame(sock, chunk.pack())


def send_layer_blocks(
    peer: tuple[str, int],
    transfer_id: str,
    request_id: str,
    layer_index: int,
    data: object,
    block_size: int,
    block_count: int,
) -> None:
    if _native_configured():
        from .tcp_native import send_layer_blocks as native_send_layer_blocks

        native_send_layer_blocks(
            peer,
            transfer_id,
            request_id,
            layer_index,
            data,
            block_size,
            block_count,
        )
        return
    _send_block_layer(
        peer,
        _BLOCK_LAYER_MAGIC,
        transfer_id,
        request_id,
        layer_index,
        data,
        block_size,
        block_count,
    )


def send_native_layer_blocks(
    peer: tuple[str, int],
    transfer_id: str,
    request_id: str,
    layer_index: int,
    data: object,
    block_size: int,
    block_count: int,
) -> None:
    if _native_configured():
        from .tcp_native import (
            send_native_layer_blocks as native_send_native_layer_blocks,
        )

        native_send_native_layer_blocks(
            peer,
            transfer_id,
            request_id,
            layer_index,
            data,
            block_size,
            block_count,
        )
        return
    _send_block_layer(
        peer,
        _NATIVE_LAYER_MAGIC,
        transfer_id,
        request_id,
        layer_index,
        data,
        block_size,
        block_count,
    )


def send_compressed_native_layer_blocks(
    peer: tuple[str, int],
    transfer_id: str,
    request_id: str,
    layer_index: int,
    data: object,
    *,
    raw_block_size: int,
    block_count: int,
    raw_bytes: int,
    codec: str,
    chunk_index: int = 0,
    chunks_in_layer: int = 1,
) -> None:
    if _native_configured() and chunks_in_layer == 1:
        from .tcp_native import (
            send_compressed_native_layer_blocks as native_send_compressed_native_layer_blocks,
        )

        native_send_compressed_native_layer_blocks(
            peer,
            transfer_id,
            request_id,
            layer_index,
            data,
            raw_block_size=raw_block_size,
            block_count=block_count,
            raw_bytes=raw_bytes,
            codec=codec,
        )
        return
    _send_compressed_native_layer(
        peer,
        transfer_id,
        request_id,
        layer_index,
        data,
        raw_block_size=raw_block_size,
        block_count=block_count,
        raw_bytes=raw_bytes,
        codec=codec,
        chunk_index=chunk_index,
        chunks_in_layer=chunks_in_layer,
    )


def send_layer_group_blocks(
    peer: tuple[str, int],
    transfer_id: str,
    request_id: str,
    first_layer: int,
    payloads: list[object],
    block_size: int,
    block_count: int,
    *,
    layout: str = "block_major",
) -> None:
    if block_size <= 0 or block_count <= 0 or not payloads:
        return
    layout_code = _LAYOUT_CODES[layout]
    views = [_payload_view(payload) for payload in payloads]
    expected = block_size * block_count
    if any(len(view) != expected for view in views):
        raise ValueError("group payload length does not match metadata")
    header = bytearray(_GROUP_LAYER_MAGIC)
    header.extend(_write_string(transfer_id))
    header.extend(_write_string(request_id))
    header.extend(_U32.pack(first_layer))
    header.extend(_U32.pack(len(views)))
    header.extend(_U32.pack(block_count))
    header.extend(_U32.pack(block_size))
    header.extend(_U32.pack(layout_code))
    with _connect(peer) as sock:
        _send_frame_parts(sock, [header, *views])


def send_kv_layer_blocks(
    peer: tuple[str, int],
    transfer_id: str,
    request_id: str,
    layer_index: int,
    key_data: object,
    value_data: object,
    part_size: int,
    block_count: int,
) -> None:
    if part_size <= 0 or block_count <= 0:
        return
    if _native_configured():
        from .tcp_native import send_kv_layer_blocks as native_send_kv_layer_blocks

        native_send_kv_layer_blocks(
            peer,
            transfer_id,
            request_id,
            layer_index,
            key_data,
            value_data,
            part_size,
            block_count,
        )
        return
    key = _payload_view(key_data)
    value = _payload_view(value_data)
    expected = part_size * block_count
    if len(key) != expected or len(value) != expected:
        raise ValueError("kv-major payload length does not match metadata")
    header = _layer_header(
        _KV_MAJOR_LAYER_MAGIC,
        transfer_id,
        request_id,
        layer_index,
        block_count,
        part_size,
    )
    with _connect(peer) as sock:
        _send_frame_parts(sock, [header, key, value])


def send_grant(
    peer: tuple[str, int],
    layer_index: int,
    kind: str = "grant",
    transfer_id: str = "",
) -> None:
    with _connect(peer) as sock:
        _send_frame(sock, Grant(layer_index, kind=kind, transfer_id=transfer_id).pack())


def _connect(peer: tuple[str, int]) -> socket.socket:
    timeout = float(os.environ.get("UNCHAIN_KV_TCP_CONNECT_TIMEOUT_S", "30"))
    io_timeout = float(os.environ.get("UNCHAIN_KV_TCP_IO_TIMEOUT_S", "300"))
    retry_s = float(os.environ.get("UNCHAIN_KV_TCP_RETRY_INTERVAL_S", "0.05"))
    deadline = time.monotonic() + timeout
    last_error: ConnectionRefusedError | None = None
    while True:
        try:
            remaining = max(0.1, deadline - time.monotonic())
            sock = socket.create_connection(peer, timeout=min(1.0, remaining))
            sock.settimeout(None if io_timeout <= 0 else io_timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except ConnectionRefusedError as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise last_error
            time.sleep(retry_s)


def _send_frame(sock: socket.socket, data: bytes) -> None:
    _send_frame_parts(sock, [data])


def _send_block_layer(
    peer: tuple[str, int],
    magic: bytes,
    transfer_id: str,
    request_id: str,
    layer_index: int,
    data: object,
    block_size: int,
    block_count: int,
) -> None:
    if block_size <= 0 or block_count <= 0:
        return
    payload = _payload_view(data)
    if len(payload) != block_size * block_count:
        raise ValueError("block payload length does not match metadata")
    header = _layer_header(
        magic, transfer_id, request_id, layer_index, block_count, block_size
    )
    with _connect(peer) as sock:
        _send_frame_parts(sock, [header, payload])


def _send_compressed_native_layer(
    peer: tuple[str, int],
    transfer_id: str,
    request_id: str,
    layer_index: int,
    data: object,
    *,
    raw_block_size: int,
    block_count: int,
    raw_bytes: int,
    codec: str,
    chunk_index: int = 0,
    chunks_in_layer: int = 1,
) -> None:
    payload = _payload_view(data)
    codec_code = _CODEC_CODES[codec]
    if raw_block_size < 0 or block_count < 0 or raw_bytes < 0:
        raise ValueError("compressed native metadata must be non-negative")
    header = bytearray(
        _COMPRESSED_NATIVE_CHUNK_MAGIC
        if chunks_in_layer > 1
        else _COMPRESSED_NATIVE_LAYER_MAGIC
    )
    header.extend(_write_string(transfer_id))
    header.extend(_write_string(request_id))
    header.extend(_U32.pack(layer_index))
    header.extend(_U32.pack(block_count))
    header.extend(_U32.pack(raw_block_size))
    header.extend(_U32.pack(raw_bytes))
    header.extend(_U32.pack(codec_code))
    if chunks_in_layer > 1:
        if not 0 <= chunk_index < chunks_in_layer:
            raise ValueError("compressed native chunk index out of range")
        header.extend(_U32.pack(chunk_index))
        header.extend(_U32.pack(chunks_in_layer))
    header.extend(_U32.pack(len(payload)))
    with _connect(peer) as sock:
        _send_frame_parts(sock, [header, payload])


def _layer_header(
    magic: bytes,
    transfer_id: str,
    request_id: str,
    layer_index: int,
    count: int,
    unit_size: int,
) -> bytearray:
    header = bytearray(magic)
    header.extend(_write_string(transfer_id))
    header.extend(_write_string(request_id))
    header.extend(_U32.pack(layer_index))
    header.extend(_U32.pack(count))
    header.extend(_U32.pack(unit_size))
    return header


def _send_frame_parts(sock: socket.socket, parts: list[object]) -> None:
    views = [_payload_view(part) for part in parts if len(part)]
    checksum = 0
    for view in views:
        checksum = crc32(view, checksum)
    sock.sendall(
        _FRAME_HEADER.pack(sum(len(view) for view in views), checksum & 0xFFFFFFFF)
    )
    if hasattr(sock, "sendmsg"):
        _sendmsg_all(sock, views)
        return
    for view in views:
        sock.sendall(view)


def _sendmsg_all(sock: socket.socket, views: list[memoryview]) -> None:
    while views:
        sent = sock.sendmsg(views)
        if sent <= 0:
            raise OSError("socket sendmsg made no progress")
        while views and sent >= len(views[0]):
            sent -= len(views[0])
            views.pop(0)
        if sent:
            views[0] = views[0][sent:]


def _payload_view(data: object) -> memoryview:
    view = memoryview(data)
    if not view.c_contiguous:
        view = memoryview(view.tobytes())
    if view.ndim != 1 or view.format != "B":
        view = view.cast("B")
    return view


def _validate_frame(data: object, checksum: int) -> None:
    if crc32(_payload_view(data)) & 0xFFFFFFFF != checksum:
        raise ValueError("tcp frame checksum mismatch")


def _native_configured() -> bool:
    return bool(os.environ.get("UNCHAIN_KV_TCP_LIB", ""))


def _starts_with(data: object, prefix: bytes) -> bool:
    view = memoryview(data).cast("B")
    return len(view) >= len(prefix) and bytes(view[: len(prefix)]) == prefix


def _recv_exact(
    sock: socket.socket, size: int, clean_eof: bool = False
) -> bytes | None:
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        read = sock.recv_into(view[offset:])
        if not read:
            if clean_eof and offset == 0:
                return None
            raise ConnectionError("truncated tcp frame")
        offset += read
    return data


def _read_string(data: object, offset: int) -> tuple[str, int]:
    size = _U32.unpack(data[offset : offset + _U32.size])[0]
    offset += _U32.size
    end = offset + size
    return bytes(data[offset:end]).decode("utf-8"), end


def _write_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return _U32.pack(len(data)) + data
