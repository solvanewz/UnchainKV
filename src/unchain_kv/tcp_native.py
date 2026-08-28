from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import threading
import weakref

from .tcp_data import TcpReceiver

_FrameCallback = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulonglong, ctypes.c_void_p
)
_tls = threading.local()
_PyBytes_AsString = ctypes.pythonapi.PyBytes_AsString
_PyBytes_AsString.argtypes = [ctypes.py_object]
_PyBytes_AsString.restype = ctypes.c_void_p


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_PyObject_GetBuffer = ctypes.pythonapi.PyObject_GetBuffer
_PyObject_GetBuffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_PyObject_GetBuffer.restype = ctypes.c_int
_PyBuffer_Release = ctypes.pythonapi.PyBuffer_Release
_PyBuffer_Release.argtypes = [ctypes.POINTER(_PyBuffer)]
_PyBuffer_Release.restype = None


class _BufferOwner:
    def __init__(self, data: object) -> None:
        self._buffer = _PyBuffer()
        self._released = True
        if _PyObject_GetBuffer(data, ctypes.byref(self._buffer), 0) != 0:
            raise BufferError("PyObject_GetBuffer failed")
        self._released = False

    @property
    def ptr(self) -> int:
        return self._buffer.buf

    @property
    def size(self) -> int:
        return self._buffer.len

    def release(self) -> None:
        if not self._released:
            _PyBuffer_Release(ctypes.byref(self._buffer))
            self._released = True


class NativeTcpUnavailable(RuntimeError):
    pass


def configured() -> bool:
    return bool(os.environ.get("UNCHAIN_KV_TCP_LIB", ""))


def load_library():
    path = os.environ.get("UNCHAIN_KV_TCP_LIB", "")
    return _load_library(path)


@lru_cache(maxsize=None)
def _load_library(path: str):
    if not path or not Path(path).exists():
        raise NativeTcpUnavailable(f"native TCP KV library not found: {path}")
    lib = ctypes.CDLL(path)
    lib.kvq_tcp_send_block_layer.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_ulonglong,
        ctypes.c_uint,
        ctypes.c_uint,
    ]
    lib.kvq_tcp_send_block_layer.restype = ctypes.c_int
    lib.kvq_tcp_send_native_layer.argtypes = lib.kvq_tcp_send_block_layer.argtypes
    lib.kvq_tcp_send_native_layer.restype = ctypes.c_int
    if hasattr(lib, "kvq_tcp_send_compressed_native_layer"):
        lib.kvq_tcp_send_compressed_native_layer.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_ulonglong,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        lib.kvq_tcp_send_compressed_native_layer.restype = ctypes.c_int
    if hasattr(lib, "kvq_tcp_connect"):
        lib.kvq_tcp_connect.argtypes = [ctypes.c_char_p]
        lib.kvq_tcp_connect.restype = ctypes.c_int
        lib.kvq_tcp_close.argtypes = [ctypes.c_int]
        lib.kvq_tcp_close.restype = ctypes.c_int
        lib.kvq_tcp_send_block_layer_fd.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_ulonglong,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        lib.kvq_tcp_send_block_layer_fd.restype = ctypes.c_int
        lib.kvq_tcp_send_native_layer_fd.argtypes = (
            lib.kvq_tcp_send_block_layer_fd.argtypes
        )
        lib.kvq_tcp_send_native_layer_fd.restype = ctypes.c_int
        if hasattr(lib, "kvq_tcp_send_compressed_native_layer_fd"):
            lib.kvq_tcp_send_compressed_native_layer_fd.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_ulonglong,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
            ]
            lib.kvq_tcp_send_compressed_native_layer_fd.restype = ctypes.c_int
    lib.kvq_tcp_send_kv_layer.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_ulonglong,
        ctypes.c_void_p,
        ctypes.c_ulonglong,
        ctypes.c_uint,
        ctypes.c_uint,
    ]
    lib.kvq_tcp_send_kv_layer.restype = ctypes.c_int
    if hasattr(lib, "kvq_tcp_send_kv_layer_fd"):
        lib.kvq_tcp_send_kv_layer_fd.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_ulonglong,
            ctypes.c_void_p,
            ctypes.c_ulonglong,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        lib.kvq_tcp_send_kv_layer_fd.restype = ctypes.c_int
    lib.kvq_tcp_start_receiver.argtypes = [
        ctypes.c_char_p,
        _FrameCallback,
        ctypes.c_void_p,
    ]
    lib.kvq_tcp_start_receiver.restype = ctypes.c_int
    lib.kvq_tcp_stop_receiver.argtypes = []
    lib.kvq_tcp_stop_receiver.restype = ctypes.c_int
    lib.kvq_tcp_free.argtypes = [ctypes.c_void_p]
    lib.kvq_tcp_free.restype = None
    lib.kvq_tcp_last_error.restype = ctypes.c_char_p
    return lib


class NativeTcpReceiver(TcpReceiver):
    def __init__(
        self,
        bind: tuple[str, int],
        store,
        trace=None,
        max_bytes: int = 128 * 1024 * 1024,
        recv_buffer_bytes: int | None = None,
        grants=None,
    ) -> None:
        del max_bytes, recv_buffer_bytes
        self.bind = bind
        self.store = store
        self.grants = grants
        self.trace = trace
        self.port = bind[1]
        self._closed = threading.Event()
        self.lib = load_library()
        self._callback = _FrameCallback(self._on_frame)

    def serve(self) -> None:
        try:
            _check(
                self.lib.kvq_tcp_start_receiver(
                    _peer(self.bind), self._callback, None
                )
            )
            self._closed.wait()
        except Exception as exc:
            if not self._closed.is_set():
                self.store.fail(exc)

    def close(self) -> None:
        self._closed.set()
        self.lib.kvq_tcp_stop_receiver()

    def _on_frame(self, data, data_len, context) -> None:
        del context
        try:
            ptr = ctypes.cast(data, ctypes.c_void_p).value
            if ptr is None:
                raise ValueError("native TCP callback received a null frame")
            owner = (ctypes.c_ubyte * data_len).from_address(ptr)
            weakref.finalize(owner, self.lib.kvq_tcp_free, ctypes.c_void_p(ptr))
            self._handle_frame(memoryview(owner).cast("B"))
        except Exception as exc:
            self.store.fail(exc)


def send_layer_blocks(
    peer: tuple[str, int],
    transfer_id: str,
    request_id: str,
    layer_index: int,
    data: object,
    block_size: int,
    block_count: int,
) -> None:
    if block_size <= 0 or block_count <= 0:
        return
    ptr, size, owner = _buffer_pointer(data)
    try:
        lib = load_library()
        _send_block(
            lib,
            peer,
            lib.kvq_tcp_send_block_layer,
            getattr(lib, "kvq_tcp_send_block_layer_fd", None),
            transfer_id,
            request_id,
            layer_index,
            ptr,
            size,
            block_size,
            block_count,
        )
    finally:
        _release_buffer_owner(owner)


def send_native_layer_blocks(
    peer: tuple[str, int],
    transfer_id: str,
    request_id: str,
    layer_index: int,
    data: object,
    block_size: int,
    block_count: int,
) -> None:
    if block_size <= 0 or block_count <= 0:
        return
    ptr, size, owner = _buffer_pointer(data)
    try:
        lib = load_library()
        _send_block(
            lib,
            peer,
            lib.kvq_tcp_send_native_layer,
            getattr(lib, "kvq_tcp_send_native_layer_fd", None),
            transfer_id,
            request_id,
            layer_index,
            ptr,
            size,
            block_size,
            block_count,
        )
    finally:
        _release_buffer_owner(owner)


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
) -> None:
    if raw_block_size <= 0 or block_count <= 0:
        return
    from .tcp_data import _CODEC_CODES, _send_compressed_native_layer

    ptr, size, owner = _buffer_pointer(data)
    try:
        lib = load_library()
        direct = getattr(lib, "kvq_tcp_send_compressed_native_layer", None)
        fd_fn = getattr(lib, "kvq_tcp_send_compressed_native_layer_fd", None)
        if direct is None:
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
            )
            return
        codec_code = _CODEC_CODES[codec]
        fd = _connection_fd(lib, peer)
        if fd is not None and fd_fn is not None:
            result = fd_fn(
                fd,
                _utf8(transfer_id),
                _utf8(request_id),
                layer_index,
                ptr,
                size,
                raw_block_size,
                block_count,
                raw_bytes,
                codec_code,
            )
            if result == 0:
                return
            _drop_connection(lib, peer)
        _check(
            direct(
                _peer(peer),
                _utf8(transfer_id),
                _utf8(request_id),
                layer_index,
                ptr,
                size,
                raw_block_size,
                block_count,
                raw_bytes,
                codec_code,
            )
        )
    finally:
        _release_buffer_owner(owner)


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
    key_ptr, key_size, key_owner = _buffer_pointer(key_data)
    value_ptr, value_size, value_owner = _buffer_pointer(value_data)
    try:
        lib = load_library()
        fd = _connection_fd(lib, peer)
        if fd is not None and hasattr(lib, "kvq_tcp_send_kv_layer_fd"):
            result = lib.kvq_tcp_send_kv_layer_fd(
                fd,
                _utf8(transfer_id),
                _utf8(request_id),
                layer_index,
                key_ptr,
                key_size,
                value_ptr,
                value_size,
                part_size,
                block_count,
            )
            if result == 0:
                return
            _drop_connection(lib, peer)
        _check(
            lib.kvq_tcp_send_kv_layer(
                _peer(peer),
                _utf8(transfer_id),
                _utf8(request_id),
                layer_index,
                key_ptr,
                key_size,
                value_ptr,
                value_size,
                part_size,
                block_count,
            )
        )
    finally:
        _release_buffer_owner(key_owner)
        _release_buffer_owner(value_owner)


def _check(result: int) -> None:
    if result == 0:
        return
    error = load_library().kvq_tcp_last_error() or b"native TCP error"
    raise RuntimeError(error.decode())


def _buffer_pointer(data: object) -> tuple[int, int, object]:
    if isinstance(data, bytes):
        return _PyBytes_AsString(data), len(data), data
    view = memoryview(data)
    if not view.c_contiguous:
        copied = view.tobytes()
        return _PyBytes_AsString(copied), len(copied), copied
    if view.ndim != 1 or view.format != "B":
        view = view.cast("B")
    if view.readonly:
        owner = _BufferOwner(view)
        return owner.ptr, owner.size, owner
    if view.nbytes == 0:
        copied = b""
        return _PyBytes_AsString(copied), 0, copied
    return ctypes.addressof(ctypes.c_char.from_buffer(view)), view.nbytes, view


def _release_buffer_owner(owner: object) -> None:
    release = getattr(owner, "release", None)
    if release is not None:
        release()


@lru_cache(maxsize=4096)
def _utf8(value: str) -> bytes:
    return value.encode("utf-8")


@lru_cache(maxsize=1024)
def _peer(peer: tuple[str, int]) -> bytes:
    return f"{peer[0]}:{peer[1]}".encode()


def _send_block(
    lib,
    peer: tuple[str, int],
    direct_fn,
    fd_fn,
    transfer_id: str,
    request_id: str,
    layer_index: int,
    ptr: int,
    size: int,
    block_size: int,
    block_count: int,
) -> None:
    fd = _connection_fd(lib, peer)
    if fd is not None and fd_fn is not None:
        result = fd_fn(
            fd,
            _utf8(transfer_id),
            _utf8(request_id),
            layer_index,
            ptr,
            size,
            block_size,
            block_count,
        )
        if result == 0:
            return
        _drop_connection(lib, peer)
    _check(
        direct_fn(
            _peer(peer),
            _utf8(transfer_id),
            _utf8(request_id),
            layer_index,
            ptr,
            size,
            block_size,
            block_count,
        )
    )


def _connection_fd(lib, peer: tuple[str, int]) -> int | None:
    connect = getattr(lib, "kvq_tcp_connect", None)
    if connect is None:
        return None
    fds = getattr(_tls, "fds", None)
    if fds is None:
        fds = {}
        _tls.fds = fds
    key = _peer(peer)
    fd = fds.get(key)
    if fd is None:
        fd = connect(key)
        if fd < 0:
            _check(fd)
        fds[key] = fd
    return fd


def _drop_connection(lib, peer: tuple[str, int]) -> None:
    fds = getattr(_tls, "fds", None)
    if not fds:
        return
    fd = fds.pop(_peer(peer), None)
    if fd is not None and hasattr(lib, "kvq_tcp_close"):
        lib.kvq_tcp_close(fd)
