from __future__ import annotations

import threading


class GrantStore:
    def __init__(self) -> None:
        self._layers: set[tuple[str, str, int]] = set()
        self._cond = threading.Condition()

    def add(self, layer_index: int, kind: str = "grant", transfer_id: str = "") -> bool:
        key = (kind, transfer_id, layer_index)
        with self._cond:
            is_new = key not in self._layers
            self._layers.add(key)
            self._cond.notify_all()
            return is_new

    def wait(
        self,
        layer_index: int,
        timeout: float,
        kind: str = "grant",
        transfer_id: str = "",
    ) -> None:
        key = (kind, transfer_id, layer_index)
        with self._cond:
            if key in self._layers:
                return
            if not self._cond.wait_for(lambda: key in self._layers, timeout):
                raise TimeoutError(
                    f"{kind} not received for layer {layer_index}"
                    + (f" transfer {transfer_id}" if transfer_id else "")
                )

    def has(
        self,
        layer_index: int,
        kind: str = "grant",
        transfer_id: str = "",
    ) -> bool:
        key = (kind, transfer_id, layer_index)
        with self._cond:
            return key in self._layers

    def wait_value(self, timeout: float, kind: str, transfer_id: str) -> int:
        def value() -> int | None:
            return next(
                (
                    layer_index
                    for item_kind, item_transfer, layer_index in self._layers
                    if item_kind == kind and item_transfer == transfer_id
                ),
                None,
            )

        with self._cond:
            if value() is None and not self._cond.wait_for(
                lambda: value() is not None, timeout
            ):
                raise TimeoutError(f"{kind} not received for transfer {transfer_id}")
            layer_index = value()
            assert layer_index is not None
            self._layers.remove((kind, transfer_id, layer_index))
            return layer_index
