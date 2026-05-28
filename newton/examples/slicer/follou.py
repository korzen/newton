# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MiniMou adapter for the vendored Follou SDK."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FOLLOU_ROOT = Path(__file__).resolve().parent / "follou_sdk"

_manager_lock = threading.Lock()
_manager_entry: "_ManagerEntry | None" = None


@dataclass
class _ManagerEntry:
    """Shared SDK scan result; new acquisitions reuse devices until the last release."""

    manager: object
    refcount: int = 0


@dataclass
class _MiniMouSampleState:
    handle_min: float | None = None
    handle_max: float | None = None
    tool_min: float | None = None
    tool_max: float | None = None


def ensure_follou_importable():
    try:
        from newton.examples.slicer.follou_sdk.devices.minimou import MiniMou
        from newton.examples.slicer.follou_sdk.manager import DeviceManager
    except Exception as exc:
        raise RuntimeError(f"Failed to import vendored Follou SDK from {FOLLOU_ROOT}: {exc}") from exc
    return DeviceManager, MiniMou


def acquire_manager():
    DeviceManager, MiniMou = ensure_follou_importable()
    global _manager_entry
    with _manager_lock:
        if _manager_entry is None:
            _manager_entry = _ManagerEntry(manager=DeviceManager())
        _manager_entry.refcount += 1
        return _manager_entry.manager, MiniMou


def release_manager() -> None:
    global _manager_entry
    with _manager_lock:
        if _manager_entry is None:
            return

        _manager_entry.refcount -= 1
        if _manager_entry.refcount > 0:
            return

        for device in getattr(_manager_entry.manager, "devices", []):
            close = getattr(device, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        _manager_entry = None


class MiniMouController:
    """Thin polling wrapper around a MiniMou SDK controller."""

    def __init__(
        self,
        *,
        device_index: int = 0,
        scale: float | tuple[float, float, float] = 1.0,
    ) -> None:
        device_index = _device_index(device_index)

        self._manager, self._mini_mou_cls = acquire_manager()
        self.device_index = device_index
        self.scale = scale
        self._closed = False
        self._sample_state = _MiniMouSampleState()

        self._controller = self._manager.get_device_controller(self._mini_mou_cls, count=self.device_index)

        if self._controller is None:
            available = [type(device).__name__ for device in getattr(self._manager, "devices", [])]
            release_manager()
            raise RuntimeError(
                f"MiniMou device index {self.device_index} not available; "
                f"discovered devices: {available or ['none']}"
            )

    def poll(self) -> dict:
        if self._closed:
            return {"active": False, "valid": False}
        return poll_minimou_controller(self._controller, self._sample_state, scale=self.scale)

    def set_power(self, enabled: bool) -> None:
        setter = getattr(self._controller, "set_power", None)
        if callable(setter):
            setter(bool(enabled))

    def is_power_on(self) -> bool | None:
        getter = getattr(self._controller, "get_power_on_status", None)
        if not callable(getter):
            return None
        try:
            return bool(getter())
        except Exception:
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        release_manager()


def poll_minimou_controller(
    controller,
    state: _MiniMouSampleState | None = None,
    *,
    scale: float | tuple[float, float, float] = 1.0,
) -> dict:
    state = state or _MiniMouSampleState()
    update = getattr(controller, "perform_update", None)
    if callable(update):
        update()

    raw_position = np.asarray(controller.get_position()[:3], dtype=np.float32)
    # MiniMou X motion is mirrored relative to the example scene axes. The
    # SDK accessor also reports Y/Z in the opposite order from this example.
    position = np.asarray((-raw_position[0], -raw_position[2], raw_position[1]), dtype=np.float32)
    position *= np.asarray(scale, dtype=np.float32)
    raw_rotation = _normalize_quaternion(controller.get_orientation())
    rotation = _swap_yz_quaternion(_map_minimou_quaternion_to_position_frame(raw_rotation))

    tool_pos = _read_float(getattr(controller, "get_tool_pos", lambda: 0.0), 0.0)
    handle_pos = _read_float(getattr(controller, "get_handle_opening_value", lambda: tool_pos), tool_pos)
    handle_active = bool(_read_float(getattr(controller, "get_handle_activity", lambda: 0.0), 0.0))
    grip = max(
        _closing_grip(handle_pos, state, "handle_min", "handle_max"),
        _closing_grip(tool_pos, state, "tool_min", "tool_max"),
    )
    button = handle_active or grip >= 0.5

    return {
        "active": True,
        "position": position,
        "rotation": rotation,
        "raw_position": raw_position,
        "raw_rotation": raw_rotation,
        "button": button,
        "grip": grip,
        "tool_scalar": tool_pos,
        "handle_pos": handle_pos,
        "handle_active": handle_active,
        "valid": True,
    }


def _read_float(reader, default: float) -> float:
    try:
        value = float(reader())
    except Exception:
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def _device_index(value) -> int:
    if isinstance(value, bool):
        raise ValueError("MiniMou device_index must be a non-negative integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lstrip("+-").isdigit() is False:
            raise ValueError("MiniMou device_index must be a non-negative integer")
        result = int(stripped)
    else:
        raise ValueError("MiniMou device_index must be a non-negative integer")
    if result < 0:
        raise ValueError("MiniMou device_index must be a non-negative integer")
    return result


def _closing_grip(value: float, state: _MiniMouSampleState, min_attr: str, max_attr: str) -> float:
    if not math.isfinite(value):
        return 0.0

    current_min = getattr(state, min_attr)
    current_max = getattr(state, max_attr)
    if current_min is None or current_max is None:
        setattr(state, min_attr, float(value))
        setattr(state, max_attr, float(value))
        return 0.0

    current_min = min(float(current_min), float(value))
    current_max = max(float(current_max), float(value))
    setattr(state, min_attr, current_min)
    setattr(state, max_attr, current_max)
    span = current_max - current_min
    if span < 1.0e-4:
        return 0.0

    opening = (float(value) - current_min) / span
    return float(np.clip(1.0 - opening, 0.0, 1.0))


def _normalize_quaternion(quaternion) -> np.ndarray:
    values = np.asarray([float(value) for value in tuple(quaternion)[:4]], dtype=np.float32)
    if values.shape[0] != 4:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if norm < 1.0e-8 or not math.isfinite(norm):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    values = values / norm
    if values[3] < 0.0:
        values = -values
    return values


def _map_minimou_quaternion_to_position_frame(rotation: np.ndarray) -> np.ndarray:
    # The Follou SDK returns a quaternion, not axis/angle. Rotate its basis into
    # the same frame as get_position() without changing the position mapping.
    half_sqrt2 = np.float32(np.sqrt(0.5))
    tet_to_hex = np.asarray((half_sqrt2, 0.0, 0.0, half_sqrt2), dtype=np.float32)
    hex_to_tet = np.asarray((-half_sqrt2, 0.0, 0.0, half_sqrt2), dtype=np.float32)
    return _quat_mul(_quat_mul(hex_to_tet, rotation), tet_to_hex)


def _swap_yz_quaternion(rotation: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(rotation)
    return _normalize_quaternion((x, -z, y, w))


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = _normalize_quaternion(left)
    rx, ry, rz, rw = _normalize_quaternion(right)
    return _normalize_quaternion(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )
