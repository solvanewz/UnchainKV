from __future__ import annotations

from dataclasses import dataclass, field
import threading

from .protocol import Chunk


@dataclass
class _Layer:
    expected: int | None
    layout: str = "block_major"
    metadata: dict[str, object] = field(default_factory=dict)
    chunks: dict[int, bytes] = field(default_factory=dict)
    ready: threading.Event = field(default_factory=threading.Event)
    released: bool = False


class LayerStore:
    def __init__(self) -> None:
        self._layers: dict[tuple[str, int], _Layer] = {}
        self._requests: dict[str, str] = {}
        # ponytail: process-lifetime tombstones; expire them only if million-request
        # service runs make this measurable.
        self._failures: dict[str, BaseException] = {}
        self._lock = threading.Lock()

    def add(
        self,
        chunk: Chunk,
        layout: str = "block_major",
        metadata: dict[str, object] | None = None,
    ) -> bool:
        header = chunk.header
        try:
            if header.layer_index < 0:
                raise ValueError("negative layer_index")
            if header.chunks_in_layer <= 0:
                raise ValueError("chunks_in_layer must be positive")
            if not 0 <= header.chunk_index < header.chunks_in_layer:
                raise ValueError("chunk_index outside layer")
            if header.payload_len != len(chunk.payload):
                raise ValueError("payload length mismatch")
        except BaseException as exc:
            self.fail(exc, header.transfer_id)
            raise
        key = (chunk.header.transfer_id, chunk.header.layer_index)
        with self._lock:
            try:
                self._raise_failure_locked(header.transfer_id)
                request_id = self._requests.setdefault(
                    header.transfer_id, header.request_id
                )
                if request_id != header.request_id:
                    raise ValueError("transfer_id reused by another request")
                layer = self._layers.setdefault(key, _Layer(None))
                if layer.released:
                    raise RuntimeError("layer payload already released")
                if not layer.chunks and layer.expected is None:
                    layer.layout = layout
                    layer.metadata = dict(metadata or {})
                elif layer.layout != layout:
                    raise ValueError("inconsistent layer layout")
                elif metadata is not None and layer.metadata != metadata:
                    raise ValueError("inconsistent layer metadata")
                if layer.expected is None:
                    layer.expected = header.chunks_in_layer
                elif layer.expected != header.chunks_in_layer:
                    raise ValueError("inconsistent chunks_in_layer")
                if header.chunk_index in layer.chunks:
                    raise ValueError(
                        "duplicate chunk_index: "
                        f"transfer={header.transfer_id} layer={header.layer_index} "
                        f"chunk={header.chunk_index}"
                    )
                layer.chunks[header.chunk_index] = chunk.payload
                if len(layer.chunks) == layer.expected:
                    layer.ready.set()
                    return True
                return False
            except BaseException as exc:
                self._fail_locked(exc, header.transfer_id)
                raise

    def is_ready(self, transfer_id: str, layer_index: int) -> bool:
        with self._lock:
            if transfer_id in self._failures:
                return False
            layer = self._layers.get((transfer_id, layer_index))
            return bool(layer and layer.ready.is_set())

    def wait(self, transfer_id: str, layer_index: int, timeout: float) -> None:
        key = (transfer_id, layer_index)
        with self._lock:
            self._raise_failure_locked(transfer_id)
            layer = self._layers.setdefault(key, _Layer(None))
        if not layer.ready.wait(timeout):
            error = TimeoutError(f"layer {layer_index} not ready for {transfer_id}")
            self.fail(error, transfer_id)
            raise error
        with self._lock:
            self._raise_failure_locked(transfer_id)

    def layer_payloads(self, transfer_id: str, layer_index: int) -> list[bytes]:
        with self._lock:
            self._raise_failure_locked(transfer_id)
            layer = self._layers[(transfer_id, layer_index)]
            if not layer.ready.is_set() or layer.released:
                raise RuntimeError("layer payload is not available")
            return [layer.chunks[index] for index in range(layer.expected or 0)]

    def layer_layout(self, transfer_id: str, layer_index: int) -> str:
        with self._lock:
            self._raise_failure_locked(transfer_id)
            return self._layers[(transfer_id, layer_index)].layout

    def layer_metadata(self, transfer_id: str, layer_index: int) -> dict[str, object]:
        with self._lock:
            self._raise_failure_locked(transfer_id)
            return dict(self._layers[(transfer_id, layer_index)].metadata)

    def release_payloads(self, transfer_id: str, layer_index: int) -> int:
        with self._lock:
            layer = self._layers.get((transfer_id, layer_index))
            if layer is None or not layer.ready.is_set() or layer.released:
                return 0
            released = sum(len(payload) for payload in layer.chunks.values())
            layer.chunks.clear()
            layer.released = True
            return released

    def discard_transfer(self, transfer_id: str) -> int:
        with self._lock:
            keys = [key for key in self._layers if key[0] == transfer_id]
            released = sum(
                len(payload)
                for key in keys
                for payload in self._layers[key].chunks.values()
            )
            for key in keys:
                self._layers[key].ready.set()
                del self._layers[key]
            self._requests.pop(transfer_id, None)
            self._failures.setdefault(
                transfer_id, RuntimeError(f"transfer {transfer_id} discarded")
            )
            return released

    def fail(
        self, error: BaseException, transfer_id: str | None = None
    ) -> int:
        with self._lock:
            return self._fail_locked(error, transfer_id)

    def _fail_locked(
        self, error: BaseException, transfer_id: str | None
    ) -> int:
        if transfer_id is None:
            transfers = set(self._requests)
            transfers.update(current for current, _layer in self._layers)
            for current in transfers:
                self._failures.setdefault(current, error)
        else:
            self._failures.setdefault(transfer_id, error)
            transfers = {transfer_id}
        released = 0
        for (current, _layer_index), layer in self._layers.items():
            if current not in transfers:
                continue
            released += sum(len(payload) for payload in layer.chunks.values())
            layer.chunks.clear()
            layer.ready.set()
        return released

    def _raise_failure_locked(self, transfer_id: str) -> None:
        error = self._failures.get(transfer_id)
        if error is not None:
            raise error

    def payload_bytes(self, transfer_id: str | None = None) -> int:
        with self._lock:
            return sum(
                len(payload)
                for (current_transfer, _), layer in self._layers.items()
                if transfer_id is None or current_transfer == transfer_id
                for payload in layer.chunks.values()
            )

    def observed_session(self) -> tuple[str, str] | None:
        with self._lock:
            for transfer_id, request_id in self._requests.items():
                if transfer_id not in self._failures:
                    return transfer_id, request_id
        return None
