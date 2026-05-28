import logging
from typing import Type
from . import ffi
from .devices import DEVICE_MAP
from typing import Type, TypeVar, Optional
from .base import DeviceController

log = logging.getLogger(__name__)

T = TypeVar("T", bound=DeviceController)

class DeviceManager:
    def __init__(self):
        n = ffi.InitDLL()
        log.info("DLL sees %d COM ports", n)
        self.devices: list[DeviceController] = []

        while True:
            device_id = ffi.GetNextDevice()
            if device_id == -1:
                break

            device_type = ffi.GetDeviceType().decode()
            port  = ffi.GetCOMPortName().decode()
            log.info("Found %s on %s (id=%d)", device_type, port, device_id)

            for key, cls in DEVICE_MAP.items():
                if key in device_type:
                    self.devices.append(cls(device_id))
                    break
            else:
                log.warning("No Python wrapper for device type '%s'", device_type)

    # helper
    def first(self, cls: Type[T]) -> Optional[T]:
        for d in self.devices:
            if isinstance(d, cls):
                return d
        return None
    

    def get_device_controller(self, cls: Type[T], count: int = 0) -> Optional[T]:
        """
        Return the *count*-th controller of the requested class *cls*
        (0-based).  If fewer than *count+1* devices of that type exist,
        return None—mirrors the behaviour of the C# version.
        """
        found = 0
        for d in self.devices:
            if isinstance(d, cls):
                if found == count:
                    return d          # type: ignore[return-value]
                found += 1
        return None