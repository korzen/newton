from __future__ import annotations
import time, logging
from dataclasses import dataclass, field
from . import ffi

log = logging.getLogger(__name__)

@dataclass
class DeviceController:
    device_id: int
    cmd_enum_size: int              # subclasses set this before super().__init__()
    divider:      list[float] = field(init=False)
    num_rets:     list[int]   = field(init=False)
    num_sends:    list[int]   = field(init=False)
    update_interval: list[int]   = field(init=False)
    get_dt_cmd:   int         = -1     # cmd index that returns Δt (if any)
    update_freq:  float       = 0.0    # kHz, smoothed
    _t0:          float       = field(default_factory=time.perf_counter)

    # ---- ctor -----------------------------------------------------------
    def __post_init__(self):
        self.divider      = [1.0]  * self.cmd_enum_size
        self.num_rets     = [0]    * self.cmd_enum_size
        self.num_sends    = [0]    * self.cmd_enum_size
        self.update_interval = [0]    * self.cmd_enum_size

    # ---- helpers --------------------------------------------------------
    def _setup_one(self, cmd: int):
        ffi.SetupCmd(self.device_id, cmd,
                     self.num_rets[cmd], self.num_sends[cmd],
                     self.divider[cmd],  self.update_interval[cmd])

    def transfer_matrix(self):
        for k in range(self.cmd_enum_size):
            self._setup_one(k)

    def perform_update(self):
        t0 = time.perf_counter()
        ffi.UpdateDevice(self.device_id)
        if self.get_dt_cmd >= 0:
            dt = max(1e-5, self.get_return(self.get_dt_cmd, 0))
            self.update_freq = 0.95 * self.update_freq + 0.05 / dt
        self._t0 = t0

    # ---- raw DLL wrappers ----------------------------------------------
    def get_return(self, cmd: int, ix: int=0) -> float:
        return ffi.GetReturnVal(self.device_id, cmd, ix)# / self.divider[cmd]

    def set_send_float(self, cmd: int, ix: int, value: float):
        ffi.SetSendVal(self.device_id, cmd, ix, int(value * self.divider[cmd]))

    def set_send_val(self, cmd: int, ix: int, value: int):
        ffi.SetSendVal(self.device_id, cmd, ix, value)

    def close(self):                             # optional tidy-up
        ffi.CloseDevice(self.device_id)