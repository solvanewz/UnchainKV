from __future__ import annotations

import ctypes
import os
from pathlib import Path


_LIB = None
_LOAD_ERROR: str | None = None
_TOP16_TABLE_CACHE = {}
_TOP16_CODEBOOKS = bytes.fromhex(
    """
7f7e7d807c81857b827a8483867978777c7d7b7a79787e7776757f74737280717f807e817d827c7b837a8479788577767b7c7a797d787776
7e7574737f7271707f807e7d817c7b827a798378777675747c7d7b7a797e7877767f7574807372717f807e7d817c7b7a8283798478857776
7c7d7b7e7a79787f77767574737280717f7e807d7c7b817a79827877767574737d7c7b7e7a79787f77768075817473727f7e807d7c7b7a81
82797877767583747d7e7c7b7f7a797880777675747372717e7f7d807c7b817a82797877767574737d7c7e7b7a7f79787780767574738172
7f7e807d7c7b817a82797877837675747d7e7c7b7f7a797877807675747372717f7e807d7c817b7a79827877767574737d7e7c7b7f7a7980
78777681757473727f7e807d7c7b817a82797877767574737d7e7c7b7f7a797877807675747372717f7e807d7c7b817a8279787776758374
7d7e7c7b7f7a797877807675747372717f7e807d7c7b817a79827877768375747d7e7c7b7f7a797880777675747381727f7e807d7c7b817a
82797877767574737e7d7c7f7b7a797877807675747372717f7e807d7c817b7a82797883778476757d7e7c7b7f7a79788077767574737271
7f7e807d7c7b817a79827877767574737d7e7c7f7b7a798078777675817473727f7e807d7c7b817a79827883777675747e7d7f7c7b7a8079
78777675747372717f7e807d7c817b7a82797877767574737e7d7c7f7b7a798078777675748173727f7e807d7c7b817a7982787776757473
7e7d7c7f7b7a798078777675747372717f7e7d807c7b817a79827877837675747e7d7f7c7b7a807978777675747381727f7e807d7c817b7a
82797877847675837e7d7f7c7b7a807978777675748173727f7e807d7c7b817a79827877767574737e7d7f7c7b7a80797877767574737281
7f7e807d7c7b817a82797877768375747e7d7f7c7b807a7978778176757473727f7e807d7c817b7a82797877767574737e7f7d7c807b7a79
78817776757473727f7e807d7c7b817a82797877767574737f7e7d807c7b817a79787776757482737f7e807d7c7b817a8279787783767574
7f7e7d807c7b817a79787776758274737f7e807d7c7b817a79827877768375747f7e807d817c7b7a79827877767574737f7e807d7c7b817a
7982787776837574807f817e7d827c7b7a797877837675747f7e7d807c7b817a828679838785847880817f7e7d827c7b7a79788377767574
7f807e7d7c817b7a82798378770076757d7c7e7b7a79787f77767580740073727f807e7d7c817b7a79828378770076757d7c7e7b7a7f797877807675740073727f807e7d7c817b827a797877830076757d7e7c7b7f7a797877807675007473727f807e7d817c7b7a82797877007675837d7e7c7b7f7a79788077767500747381
"""
)


def _candidate_paths() -> list[str]:
    env = os.environ.get("UNCHAIN_KV_SPLITZIP_LIB")
    if env:
        return [env]
    root = Path(__file__).resolve().parents[2]
    return [
        str(root / "native/unchain_kv/build/libunchain_kv_splitzip_cuda.so"),
        str(root / "native/unchain_kv/build-user/libunchain_kv_splitzip_cuda.so"),
        "libunchain_kv_splitzip_cuda.so",
    ]


def _load():
    global _LIB, _LOAD_ERROR
    if _LIB is not None:
        return _LIB
    errors = []
    for path in _candidate_paths():
        try:
            lib = ctypes.CDLL(path)
            lib.kvq_splitzip_bf16_encode.argtypes = [
                ctypes.c_ulonglong,
                ctypes.c_ulonglong,
                ctypes.c_ulonglong,
                ctypes.c_ulonglong,
                ctypes.c_ulonglong,
            ]
            lib.kvq_splitzip_bf16_encode.restype = ctypes.c_longlong
            try:
                lib.kvq_splitzip_bf16_encode_mode.argtypes = [
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_int,
                ]
                lib.kvq_splitzip_bf16_encode_mode.restype = ctypes.c_longlong
            except AttributeError:
                pass
            try:
                lib.kvq_splitzip_bf16_decode_fixed6.argtypes = [
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                ]
                lib.kvq_splitzip_bf16_decode_fixed6.restype = ctypes.c_longlong
                lib.kvq_splitzip_bf16_decode_fixed6_blocks.argtypes = [
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                ]
                lib.kvq_splitzip_bf16_decode_fixed6_blocks.restype = (
                    ctypes.c_longlong
                )
            except AttributeError:
                pass
            try:
                lib.kvq_splitzip_bf16_encode_top16.argtypes = [
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                ]
                lib.kvq_splitzip_bf16_encode_top16.restype = ctypes.c_longlong
                lib.kvq_splitzip_bf16_decode_top16.argtypes = [
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                ]
                lib.kvq_splitzip_bf16_decode_top16.restype = ctypes.c_longlong
                lib.kvq_splitzip_bf16_decode_top16_blocks.argtypes = [
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                    ctypes.c_ulonglong,
                ]
                lib.kvq_splitzip_bf16_decode_top16_blocks.restype = (
                    ctypes.c_longlong
                )
            except AttributeError:
                pass
            lib.kvq_splitzip_last_error.argtypes = []
            lib.kvq_splitzip_last_error.restype = ctypes.c_char_p
            _LIB = lib
            return lib
        except OSError as exc:
            errors.append(str(exc))
    _LOAD_ERROR = "; ".join(errors[-2:])
    return None


def available() -> bool:
    return _load() is not None


def encode_bf16(source, out, bits: int = 6) -> int | None:
    lib = _load()
    if lib is None:
        return None
    if getattr(getattr(source, "device", None), "type", "") != "cuda":
        return None
    if bits not in {5, 6}:
        raise ValueError("splitzip bf16 bits must be 5 or 6")
    import torch

    stream = torch.cuda.current_stream(source.device).cuda_stream
    encode_mode = getattr(lib, "kvq_splitzip_bf16_encode_mode", None)
    if encode_mode is not None:
        encoded = encode_mode(
            int(source.data_ptr()),
            int(out.data_ptr()),
            int(source.numel()),
            int(out.numel()),
            int(stream),
            int(bits),
        )
    elif bits == 6:
        encoded = lib.kvq_splitzip_bf16_encode(
            int(source.data_ptr()),
            int(out.data_ptr()),
            int(source.numel()),
            int(out.numel()),
            int(stream),
        )
    else:
        return None
    if encoded < 0:
        error = lib.kvq_splitzip_last_error()
        raise RuntimeError(error.decode() if error else "splitzip cuda encode failed")
    return int(encoded)


def _decode_layout(source, out, block_ids, raw_bytes, torch):
    if not block_ids or raw_bytes <= 0 or raw_bytes % 2:
        return None
    if getattr(getattr(source, "device", None), "type", "") != "cuda":
        return None
    if source.device != getattr(out, "device", None):
        return None
    if source.dtype != torch.uint8 or out.dtype != torch.bfloat16:
        return None
    if not source.is_contiguous() or not out.is_contiguous():
        return None
    if out.dim() < 3 or out.shape[0] != 2:
        return None
    if len(set(block_ids)) != len(block_ids) or any(
        block < 0 or block >= out.shape[1] for block in block_ids
    ):
        return None
    start = block_ids[0]
    contiguous = block_ids == list(range(start, start + len(block_ids)))
    count_words = raw_bytes // 2
    divisor = 2 * len(block_ids)
    if count_words % divisor:
        return None
    block_words = count_words // divisor
    current = torch.cuda.current_stream(out.device)
    block_ids_tensor = None
    if not contiguous:
        block_ids_tensor = torch.tensor(
            block_ids, dtype=torch.int64, device=out.device
        )
    return count_words, block_words, start, current, block_ids_tensor


def decode_fixed6(source, out, block_ids: list[int], raw_bytes: int) -> int | None:
    lib = _load()
    decode = getattr(lib, "kvq_splitzip_bf16_decode_fixed6", None) if lib else None
    if decode is None:
        return None
    import torch

    layout = _decode_layout(source, out, block_ids, raw_bytes, torch)
    if layout is None:
        return None
    count_words, block_words, start, current, block_ids_tensor = layout
    decode_blocks = getattr(lib, "kvq_splitzip_bf16_decode_fixed6_blocks", None)
    if block_ids_tensor is not None and decode_blocks is None:
        return None
    copied = (
        decode_blocks(
            int(source.data_ptr()),
            int(source.numel()),
            int(out.data_ptr()),
            count_words,
            len(block_ids),
            block_words,
            int(out.stride(0)),
            int(out.stride(1)),
            start,
            int(block_ids_tensor.data_ptr()) if block_ids_tensor is not None else 0,
            int(current.cuda_stream),
        )
        if decode_blocks is not None
        else decode(
            int(source.data_ptr()),
            int(source.numel()),
            int(out.data_ptr()),
            count_words,
            len(block_ids),
            block_words,
            int(out.stride(0)),
            int(out.stride(1)),
            start,
            int(current.cuda_stream),
        )
    )
    if block_ids_tensor is not None:
        block_ids_tensor.record_stream(current)
    if copied < 0:
        error = lib.kvq_splitzip_last_error()
        raise RuntimeError(error.decode() if error else "splitzip cuda decode failed")
    return int(copied) or None


def _top16_tables(layer_index: int, device, torch):
    if not 0 <= layer_index < len(_TOP16_CODEBOOKS) // 32:
        return None
    key = (str(device), layer_index)
    cached = _TOP16_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    codebooks = _TOP16_CODEBOOKS[layer_index * 32 : (layer_index + 1) * 32]
    values = bytearray([255]) * 512
    for plane in range(2):
        for code, exponent in enumerate(codebooks[plane * 16 : (plane + 1) * 16]):
            values[plane * 256 + exponent] = code
    values.extend(codebooks)
    cached = torch.tensor(values, dtype=torch.uint8, device=device)
    _TOP16_TABLE_CACHE[key] = cached
    return cached


def encode_top16(source, out, layer_index: int) -> int | None:
    lib = _load()
    encode = getattr(lib, "kvq_splitzip_bf16_encode_top16", None) if lib else None
    if encode is None:
        return None
    if getattr(getattr(source, "device", None), "type", "") != "cuda":
        return None
    if source.device != getattr(out, "device", None):
        return None
    import torch

    if source.dtype != torch.bfloat16 or out.dtype != torch.uint8:
        return None
    if not source.is_contiguous() or not out.is_contiguous():
        return None
    tables = _top16_tables(layer_index, source.device, torch)
    if tables is None:
        return None
    stream = torch.cuda.current_stream(source.device).cuda_stream
    encoded = encode(
        int(source.data_ptr()),
        int(out.data_ptr()),
        int(source.numel()),
        int(out.numel()),
        int(tables.data_ptr()),
        int(stream),
    )
    if encoded < 0:
        error = lib.kvq_splitzip_last_error()
        raise RuntimeError(error.decode() if error else "splitzip top16 encode failed")
    return int(encoded) or None


def decode_top16(
    source, out, block_ids: list[int], raw_bytes: int, layer_index: int
) -> int | None:
    lib = _load()
    decode = getattr(lib, "kvq_splitzip_bf16_decode_top16", None) if lib else None
    if decode is None:
        return None
    import torch

    layout = _decode_layout(source, out, block_ids, raw_bytes, torch)
    if layout is None:
        return None
    tables = _top16_tables(layer_index, source.device, torch)
    if tables is None:
        return None
    count_words, block_words, start, current, block_ids_tensor = layout
    decode_blocks = getattr(lib, "kvq_splitzip_bf16_decode_top16_blocks", None)
    if block_ids_tensor is not None and decode_blocks is None:
        return None
    copied = (
        decode_blocks(
            int(source.data_ptr()),
            int(source.numel()),
            int(out.data_ptr()),
            count_words,
            len(block_ids),
            block_words,
            int(out.stride(0)),
            int(out.stride(1)),
            start,
            int(block_ids_tensor.data_ptr()) if block_ids_tensor is not None else 0,
            int(tables.data_ptr()),
            int(current.cuda_stream),
        )
        if decode_blocks is not None
        else decode(
            int(source.data_ptr()),
            int(source.numel()),
            int(out.data_ptr()),
            count_words,
            len(block_ids),
            block_words,
            int(out.stride(0)),
            int(out.stride(1)),
            start,
            int(tables.data_ptr()),
            int(current.cuda_stream),
        )
    )
    if block_ids_tensor is not None:
        block_ids_tensor.record_stream(current)
    if copied < 0:
        error = lib.kvq_splitzip_last_error()
        raise RuntimeError(error.decode() if error else "splitzip top16 decode failed")
    return int(copied) or None


def load_error() -> str | None:
    _load()
    return _LOAD_ERROR
