from enum import IntEnum
import math

from ..base import DeviceController


class Cmd(IntEnum):
    """Command table for the Scope device.

    Values copied from the original Unity enum so that they match the firmware.
    """

    RESET = 0
    GET_IDENTITY = 1
    GET_BUTTON_STATES = 2
    GET_ZOOM_LEVEL = 3
    GET_CAMERA_ANGLE = 4
    GET_CRC_POLY = 5
    GET_CURRENT_DELTA_T = 6
    GET_BUILD_DATE = 7
    SET_CAMERA_ZERO = 8


class Scope(DeviceController):
    """Python wrapper for the Follou Scope device."""

    _FLOAT_DIV = 10_000.0

    # ------------------------------------------------------------------
    def __init__(self, device_id: int):
        super().__init__(device_id, cmd_enum_size=len(Cmd))
        self.get_dt_cmd = Cmd.GET_CURRENT_DELTA_T
        self._init_matrix()
        self.transfer_matrix()

    # ------------------------------------------------------------------
    #   Command‑matrix setup
    # ------------------------------------------------------------------
    def _init_matrix(self) -> None:
        ui = self.update_interval
        nr = self.num_rets
        ns = self.num_sends  # currently unused – kept for completeness
        dv = self.divider
        fd = self._FLOAT_DIV

        # 1) subscription intervals (prime numbers to spread load)
        ui[Cmd.GET_BUTTON_STATES] = 1
        ui[Cmd.GET_ZOOM_LEVEL] = 1
        ui[Cmd.GET_CAMERA_ANGLE] = 1
        ui[Cmd.GET_CURRENT_DELTA_T] = 983

        # 2) return-value counts (GETs)
        nr[Cmd.RESET] = 1
        nr[Cmd.GET_IDENTITY] = 1
        nr[Cmd.GET_BUTTON_STATES] = 3
        nr[Cmd.GET_ZOOM_LEVEL] = 1
        nr[Cmd.GET_CAMERA_ANGLE] = 1
        nr[Cmd.GET_CRC_POLY] = 1
        nr[Cmd.GET_CURRENT_DELTA_T] = 1
        # GET_BUILD_DATE implicitly single value; not used in Unity logic
        nr[Cmd.GET_BUILD_DATE] = 1

        # 3) send-value counts (SETs)
        # Only SET_CAMERA_ZERO, which takes no payload (= implicit 0)
        ns[Cmd.SET_CAMERA_ZERO] = 0

        # 4) scaling / dividers
        dv[Cmd.GET_CAMERA_ANGLE] = fd
        dv[Cmd.GET_CURRENT_DELTA_T] = fd  # overwritten below

        # Δt sent as seconds ×100
        dv[Cmd.GET_CURRENT_DELTA_T] = 100.0

    # ------------------------------------------------------------------
    #   Public getters / helpers
    # ------------------------------------------------------------------
    def get_button_state(self, button: int) -> bool:
        """Return **True** if button *button* is *inactive* (Unity comparison `== 0`)."""
        if 0 <= button < 3:
            return self.get_return(Cmd.GET_BUTTON_STATES, button) == 0
        return False

    def get_zoom_level(self) -> int:
        return int(self.get_return(Cmd.GET_ZOOM_LEVEL, 0))

    def get_camera_angle_deg(self) -> float:
        """Return camera angle in **degrees** (firmware value is rad)."""
        return math.degrees(self.get_return(Cmd.GET_CAMERA_ANGLE, 0))

    # ------------------------------------------------------------------
    #   Commands
    # ------------------------------------------------------------------
    def set_camera_zero(self) -> None:
        """Tell the scope to store the current camera angle as zero."""
        self.set_send_val(Cmd.SET_CAMERA_ZERO, 0, 0)  # payloadless command


__all__ = [
    "Cmd",
    "ScopeDeviceController",
]
