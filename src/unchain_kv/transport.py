from __future__ import annotations

from collections.abc import Iterable
import os

from .grant_state import GrantStore
from .layer_state import LayerStore
from .protocol import Chunk
from .tcp_data import TcpReceiver
from .tcp_data import send_chunks as _tcp_send_chunks
from .tcp_data import send_grant as _tcp_send_grant


def transport_name() -> str:
    name = os.environ.get("UNCHAIN_KV_TRANSPORT", "tcp").strip().lower() or "tcp"
    if name != "tcp":
        raise ValueError(f"unsupported transport: {name}")
    return name


def make_receiver(
    bind: tuple[str, int],
    store: LayerStore,
    trace=None,
    max_bytes: int = 65535,
    recv_buffer_bytes: int | None = None,
    grants: GrantStore | None = None,
):
    transport_name()
    if os.environ.get("UNCHAIN_KV_TCP_LIB", ""):
        from .tcp_native import NativeTcpReceiver

        return NativeTcpReceiver(
            bind,
            store,
            trace=trace,
            max_bytes=max(max_bytes, 128 * 1024 * 1024),
            recv_buffer_bytes=recv_buffer_bytes,
            grants=grants,
        )
    return TcpReceiver(
        bind,
        store,
        trace=trace,
        max_bytes=max(max_bytes, 128 * 1024 * 1024),
        recv_buffer_bytes=recv_buffer_bytes,
        grants=grants,
    )


def send_chunks(peer: tuple[str, int], chunks: Iterable[Chunk]) -> None:
    transport_name()
    _tcp_send_chunks(peer, chunks)


def send_grant(
    peer: tuple[str, int],
    layer_index: int,
    kind: str = "grant",
    transfer_id: str = "",
) -> None:
    transport_name()
    _tcp_send_grant(peer, layer_index, kind=kind, transfer_id=transfer_id)
